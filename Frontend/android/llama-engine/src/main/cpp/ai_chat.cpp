#include <android/log.h>
#include <jni.h>
#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <cmath>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sampling.h>

#include "logging.h"
#include "chat.h"
#include "common.h"
#include "llama.h"

template<class T>
static std::string join(const std::vector<T> &values, const std::string &delim) {
    std::ostringstream str;
    for (size_t i = 0; i < values.size(); i++) {
        str << values[i];
        if (i < values.size() - 1) { str << delim; }
    }
    return str.str();
}

/**
 * LLama resources: context, model, batch and sampler
 */
constexpr int   N_THREADS_MIN           = 2;
constexpr int   N_THREADS_MAX           = 4;
constexpr int   N_THREADS_HEADROOM      = 2;

constexpr int   DEFAULT_CONTEXT_SIZE    = 4096;
constexpr int   OVERFLOW_HEADROOM       = 4;
constexpr int   BATCH_SIZE              = 512;
constexpr float DEFAULT_SAMPLER_TEMP    = 0.3f;

static llama_model                      * g_model;
static llama_context                    * g_context;
static llama_batch                        g_batch;
static common_chat_templates_ptr          g_chat_templates;
static common_sampler                   * g_sampler;
static int                                g_context_size = DEFAULT_CONTEXT_SIZE;
static float                              g_sampler_temp = DEFAULT_SAMPLER_TEMP;
static std::mutex                         g_diagnostics_mutex;
static std::string                       g_native_diagnostics;
static std::string                       g_native_lib_dir;
static std::mutex                         g_backend_bootstrap_mutex;
static std::vector<void *>                g_backend_library_handles;
static std::string                        g_selected_backend_library;
static bool                               g_llama_backend_initialized = false;

constexpr size_t MAX_NATIVE_DIAGNOSTICS_BYTES = 131072;

using aichat_backend_init_fn = ggml_backend_reg_t (*)();
using aichat_backend_score_fn = int (*)();

static const std::vector<std::string> &core_backend_preload_libraries() {
    static const std::vector<std::string> libraries = {
            "libggml-base.so",
            "libggml.so",
            "libllama.so",
            "libomp.so",
    };
    return libraries;
}

static const std::vector<std::string> &cpu_backend_candidate_libraries() {
    static const std::vector<std::string> libraries = {
            "libggml-cpu-android_armv9.2_2.so",
            "libggml-cpu-android_armv9.2_1.so",
            "libggml-cpu-android_armv9.0_1.so",
            "libggml-cpu-android_armv8.6_1.so",
            "libggml-cpu-android_armv8.2_2.so",
            "libggml-cpu-android_armv8.2_1.so",
            "libggml-cpu-android_armv8.0_1.so",
    };
    return libraries;
}

static std::string trim_copy(const std::string &value) {
    const auto begin = value.find_first_not_of(" \n\r\t");
    if (begin == std::string::npos) {
        return "";
    }

    const auto end = value.find_last_not_of(" \n\r\t");
    return value.substr(begin, end - begin + 1);
}

static std::string bool_to_string(const bool value) {
    return value ? "true" : "false";
}

static std::string dlerror_or_default() {
    const auto *error = dlerror();
    return error == nullptr ? "<none>" : error;
}

static void remember_backend_library_handle(void *handle) {
    if (handle == nullptr) {
        return;
    }

    if (std::find(g_backend_library_handles.begin(), g_backend_library_handles.end(), handle) ==
        g_backend_library_handles.end()) {
        g_backend_library_handles.push_back(handle);
    }
}

static void append_native_diagnostics_locked(const char *text) {
    if (text == nullptr || *text == '\0') {
        return;
    }

    g_native_diagnostics.append(text);
    if (g_native_diagnostics.size() > MAX_NATIVE_DIAGNOSTICS_BYTES) {
        g_native_diagnostics.erase(0, g_native_diagnostics.size() - MAX_NATIVE_DIAGNOSTICS_BYTES);
        const auto newline = g_native_diagnostics.find('\n');
        if (newline != std::string::npos) {
            g_native_diagnostics.erase(0, newline + 1);
        }
    }
}

static void append_native_diagnostics(const char *text) {
    std::lock_guard<std::mutex> lock(g_diagnostics_mutex);
    append_native_diagnostics_locked(text);
}

static void append_native_diagnostics(const std::string &text) {
    append_native_diagnostics(text.c_str());
}

static void append_native_diagnostics_line(const std::string &label, const std::string &value) {
    append_native_diagnostics(label + ": " + value + "\n");
}

static void append_native_diagnostics_section(const std::string &title) {
    append_native_diagnostics("\n[" + title + "]\n");
}

static std::string format_errno_value(const int errnum) {
    std::ostringstream output;
    output << errnum;
    if (errnum != 0) {
        output << " (" << std::strerror(errnum) << ")";
    }
    return output.str();
}

static std::string format_mode_octal(const mode_t mode) {
    std::ostringstream output;
    output << "0" << std::oct << (mode & 0777);
    return output.str();
}

static uint32_t read_u32_le(const unsigned char *bytes) {
    return static_cast<uint32_t>(bytes[0]) |
           (static_cast<uint32_t>(bytes[1]) << 8) |
           (static_cast<uint32_t>(bytes[2]) << 16) |
           (static_cast<uint32_t>(bytes[3]) << 24);
}

static std::string bytes_to_hex(const unsigned char *bytes, const size_t count) {
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (size_t i = 0; i < count; ++i) {
        if (i > 0) {
            output << ' ';
        }
        output << std::setw(2) << static_cast<unsigned int>(bytes[i]);
    }
    return output.str();
}

static std::string bytes_to_ascii_preview(const unsigned char *bytes, const size_t count) {
    std::string preview;
    preview.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const auto ch = static_cast<unsigned char>(bytes[i]);
        preview.push_back(std::isprint(ch) ? static_cast<char>(ch) : '.');
    }
    return preview;
}

static bool is_relevant_native_library(const std::string &file_name) {
    return file_name.rfind("libggml", 0) == 0 ||
           file_name.rfind("libllama", 0) == 0 ||
           file_name.rfind("libomp", 0) == 0 ||
           file_name.rfind("libai-chat", 0) == 0 ||
           file_name.rfind("libmtmd", 0) == 0;
}

static void append_backend_registry_snapshot(const std::string &section_name) {
    append_native_diagnostics_section(section_name);
    const auto backend_count = ggml_backend_reg_count();
    append_native_diagnostics_line("backend_registry_count", std::to_string(backend_count));
    append_native_diagnostics_line("llama_max_devices", std::to_string(llama_max_devices()));

    if (backend_count == 0) {
        append_native_diagnostics_line("registered_backends", "none");
        return;
    }

    for (size_t i = 0; i < backend_count; ++i) {
        auto *reg = ggml_backend_reg_get(i);
        append_native_diagnostics_line(
                "backend[" + std::to_string(i) + "]",
                ggml_backend_reg_name(reg));
    }
}

static void append_native_library_dir_snapshot() {
    append_native_diagnostics_section("native_library_dir");
    append_native_diagnostics_line(
            "path",
            g_native_lib_dir.empty() ? "<unknown>" : g_native_lib_dir);

    if (g_native_lib_dir.empty()) {
        return;
    }

    errno = 0;
    DIR *dir = opendir(g_native_lib_dir.c_str());
    if (dir == nullptr) {
        append_native_diagnostics_line("open_error", format_errno_value(errno));
        return;
    }

    std::vector<std::string> libraries;
    while (auto *entry = readdir(dir)) {
        const std::string file_name = entry->d_name;
        if (file_name == "." || file_name == ".." || !is_relevant_native_library(file_name)) {
            continue;
        }

        const auto full_path = g_native_lib_dir + "/" + file_name;
        struct stat file_stat {};
        if (stat(full_path.c_str(), &file_stat) == 0) {
            libraries.push_back(file_name + " (" + std::to_string(static_cast<long long>(file_stat.st_size)) + " B)");
        } else {
            libraries.push_back(file_name + " (stat failed: " + format_errno_value(errno) + ")");
        }
    }

    const int directory_error = errno;
    closedir(dir);

    if (directory_error != 0) {
        append_native_diagnostics_line("readdir_error", format_errno_value(directory_error));
    }

    std::sort(libraries.begin(), libraries.end());
    if (libraries.empty()) {
        append_native_diagnostics_line("relevant_libraries", "none");
        return;
    }

    for (size_t i = 0; i < libraries.size(); ++i) {
        append_native_diagnostics_line("library[" + std::to_string(i) + "]", libraries[i]);
    }
}

static void append_native_runtime_snapshot(const std::string &phase) {
    append_native_diagnostics_section("native_runtime_" + phase);
    append_native_diagnostics_line("supports_mmap", bool_to_string(llama_supports_mmap()));
    append_native_diagnostics_line("supports_mlock", bool_to_string(llama_supports_mlock()));
    append_native_diagnostics_line("cpu_cores_online", std::to_string(sysconf(_SC_NPROCESSORS_ONLN)));
    append_native_diagnostics_line("page_size_bytes", std::to_string(sysconf(_SC_PAGESIZE)));

    struct sysinfo memory_info {};
    if (sysinfo(&memory_info) == 0) {
        const auto unit = static_cast<unsigned long long>(memory_info.mem_unit == 0 ? 1 : memory_info.mem_unit);
        append_native_diagnostics_line(
                "total_ram_bytes",
                std::to_string(static_cast<unsigned long long>(memory_info.totalram) * unit));
        append_native_diagnostics_line(
                "free_ram_bytes",
                std::to_string(static_cast<unsigned long long>(memory_info.freeram) * unit));
    } else {
        append_native_diagnostics_line("sysinfo_error", format_errno_value(errno));
    }

    const auto system_info = trim_copy(llama_print_system_info());
    if (!system_info.empty()) {
        append_native_diagnostics("system_info:\n" + system_info + "\n");
    }

    append_backend_registry_snapshot("backend_registry_" + phase);
    append_native_library_dir_snapshot();
}

static bool bootstrap_backends_locked(
        const std::string &phase,
        const bool allow_legacy_path_scan) {
    append_native_diagnostics_section("backend_bootstrap_" + phase);
    append_native_diagnostics_line("registry_count_before", std::to_string(ggml_backend_reg_count()));
    append_native_diagnostics_line("legacy_path_scan_allowed", bool_to_string(allow_legacy_path_scan));

    if (ggml_backend_reg_count() == 0) {
        append_native_diagnostics_section("backend_preload_" + phase);
        const auto &preload_libraries = core_backend_preload_libraries();
        for (size_t i = 0; i < preload_libraries.size(); ++i) {
            const auto &soname = preload_libraries[i];
            const auto prefix = "preload[" + std::to_string(i) + "]";
            append_native_diagnostics_line(prefix + ".soname", soname);
            dlerror();
            void *handle = dlopen(soname.c_str(), RTLD_NOW | RTLD_GLOBAL);
            if (handle == nullptr) {
                append_native_diagnostics_line(prefix + ".dlopen", "failed");
                append_native_diagnostics_line(prefix + ".dlerror", dlerror_or_default());
                continue;
            }

            remember_backend_library_handle(handle);
            append_native_diagnostics_line(prefix + ".dlopen", "ok");
        }
    } else {
        append_native_diagnostics_line("preload_skipped", "registry already populated");
    }

    if (ggml_backend_reg_count() == 0) {
        append_native_diagnostics_section("backend_candidate_probe_" + phase);
        const auto &candidates = cpu_backend_candidate_libraries();
        for (size_t i = 0; i < candidates.size(); ++i) {
            const auto &soname = candidates[i];
            const auto prefix = "candidate[" + std::to_string(i) + "]";
            append_native_diagnostics_line(prefix + ".soname", soname);

            dlerror();
            void *handle = dlopen(soname.c_str(), RTLD_NOW | RTLD_GLOBAL);
            if (handle == nullptr) {
                append_native_diagnostics_line(prefix + ".dlopen", "failed");
                append_native_diagnostics_line(prefix + ".dlerror", dlerror_or_default());
                continue;
            }

            remember_backend_library_handle(handle);
            append_native_diagnostics_line(prefix + ".dlopen", "ok");

            dlerror();
            auto *score_fn = reinterpret_cast<aichat_backend_score_fn>(dlsym(handle, "ggml_backend_score"));
            const auto score_symbol_error = dlerror_or_default();
            if (score_fn == nullptr || score_symbol_error != "<none>") {
                append_native_diagnostics_line(prefix + ".score_symbol", "missing");
                append_native_diagnostics_line(prefix + ".score_symbol_error", score_symbol_error);
                continue;
            }

            const int score = score_fn();
            append_native_diagnostics_line(prefix + ".score", std::to_string(score));
            if (score <= 0) {
                append_native_diagnostics_line(prefix + ".selected", "false");
                continue;
            }

            dlerror();
            auto *init_fn = reinterpret_cast<aichat_backend_init_fn>(dlsym(handle, "ggml_backend_init"));
            const auto init_symbol_error = dlerror_or_default();
            if (init_fn == nullptr || init_symbol_error != "<none>") {
                append_native_diagnostics_line(prefix + ".init_symbol", "missing");
                append_native_diagnostics_line(prefix + ".init_symbol_error", init_symbol_error);
                continue;
            }

            auto *reg = init_fn();
            if (reg == nullptr) {
                append_native_diagnostics_line(prefix + ".init_result", "null");
                continue;
            }

            const auto backend_name = std::string(ggml_backend_reg_name(reg));
            if (ggml_backend_reg_by_name(backend_name.c_str()) == nullptr) {
                ggml_backend_register(reg);
                append_native_diagnostics_line(prefix + ".register", "ok");
            } else {
                append_native_diagnostics_line(prefix + ".register", "already_registered");
            }

            g_selected_backend_library = soname;
            append_native_diagnostics_line(prefix + ".registered_backend", backend_name);
            append_native_diagnostics_line(
                    prefix + ".registry_count_after",
                    std::to_string(ggml_backend_reg_count()));
            break;
        }
    }

    bool legacy_path_scan_ran = false;
    if (ggml_backend_reg_count() == 0 && allow_legacy_path_scan) {
        legacy_path_scan_ran = true;
        append_native_diagnostics_section("backend_legacy_path_scan_" + phase);
        append_native_diagnostics_line(
                "path",
                g_native_lib_dir.empty() ? "<unknown>" : g_native_lib_dir);
        append_native_diagnostics_line("registry_count_before_scan", std::to_string(ggml_backend_reg_count()));
        ggml_backend_load_all_from_path(g_native_lib_dir.empty() ? nullptr : g_native_lib_dir.c_str());
        append_native_diagnostics_line("registry_count_after_scan", std::to_string(ggml_backend_reg_count()));
        if (ggml_backend_reg_count() > 0 && g_selected_backend_library.empty()) {
            g_selected_backend_library = "<legacy-path-scan>";
        }
    }
    append_native_diagnostics_line("legacy_path_scan_ran", bool_to_string(legacy_path_scan_ran));

    if (ggml_backend_reg_count() > 0) {
        if (!g_llama_backend_initialized) {
            llama_backend_init();
            g_llama_backend_initialized = true;
            append_native_diagnostics_line("llama_backend_init", "called");
        } else {
            append_native_diagnostics_line("llama_backend_init", "already_initialized");
        }
        append_native_diagnostics_line(
                "selected_backend_library",
                g_selected_backend_library.empty() ? "<registered>" : g_selected_backend_library);
    } else {
        append_native_diagnostics_line("llama_backend_init", "skipped");
        append_native_diagnostics_line("selected_backend_library", "<none>");
    }

    append_native_diagnostics_line("registry_count_after", std::to_string(ggml_backend_reg_count()));
    return ggml_backend_reg_count() > 0;
}

static bool ensure_backends_ready(
        const std::string &phase,
        const bool allow_legacy_path_scan) {
    std::lock_guard<std::mutex> lock(g_backend_bootstrap_mutex);
    return bootstrap_backends_locked(phase, allow_legacy_path_scan);
}

static void append_model_file_preflight(const std::string &model_path) {
    append_native_diagnostics_section("model_file_preflight");
    append_native_diagnostics_line("path", model_path);
    append_native_diagnostics_line("exists", bool_to_string(access(model_path.c_str(), F_OK) == 0));
    append_native_diagnostics_line("readable", bool_to_string(access(model_path.c_str(), R_OK) == 0));

    struct stat file_stat {};
    if (stat(model_path.c_str(), &file_stat) != 0) {
        append_native_diagnostics_line("stat_error", format_errno_value(errno));
        return;
    }

    append_native_diagnostics_line("is_regular_file", bool_to_string(S_ISREG(file_stat.st_mode)));
    append_native_diagnostics_line("size_bytes", std::to_string(static_cast<long long>(file_stat.st_size)));
    append_native_diagnostics_line("mode_octal", format_mode_octal(file_stat.st_mode));

    errno = 0;
    FILE *file = std::fopen(model_path.c_str(), "rb");
    if (file == nullptr) {
        append_native_diagnostics_line("fopen_error", format_errno_value(errno));
        return;
    }

    unsigned char header[32] = {0};
    const size_t bytes_read = std::fread(header, 1, sizeof(header), file);
    const int read_error = std::ferror(file) ? errno : 0;
    std::fclose(file);

    append_native_diagnostics_line("header_bytes_read", std::to_string(bytes_read));
    if (bytes_read > 0) {
        append_native_diagnostics_line("header_hex", bytes_to_hex(header, bytes_read));
        append_native_diagnostics_line(
                "header_ascii_preview",
                bytes_to_ascii_preview(header, std::min<size_t>(bytes_read, 16)));
    }
    if (read_error != 0) {
        append_native_diagnostics_line("fread_error", format_errno_value(read_error));
    }

    if (bytes_read >= 4) {
        append_native_diagnostics_line(
                "gguf_magic",
                std::memcmp(header, GGUF_MAGIC, 4) == 0 ? "valid" : "invalid");
    }
    if (bytes_read >= 8) {
        append_native_diagnostics_line(
                "gguf_version_from_header",
                std::to_string(read_u32_le(header + 4)));
    }
}

static std::string gguf_value_to_string(const gguf_context *metadata, const int64_t key_id) {
    const auto value_type = gguf_get_kv_type(metadata, key_id);
    switch (value_type) {
        case GGUF_TYPE_STRING:
            return gguf_get_val_str(metadata, key_id);
        case GGUF_TYPE_UINT8:
            return std::to_string(gguf_get_val_u8(metadata, key_id));
        case GGUF_TYPE_INT8:
            return std::to_string(gguf_get_val_i8(metadata, key_id));
        case GGUF_TYPE_UINT16:
            return std::to_string(gguf_get_val_u16(metadata, key_id));
        case GGUF_TYPE_INT16:
            return std::to_string(gguf_get_val_i16(metadata, key_id));
        case GGUF_TYPE_UINT32:
            return std::to_string(gguf_get_val_u32(metadata, key_id));
        case GGUF_TYPE_INT32:
            return std::to_string(gguf_get_val_i32(metadata, key_id));
        case GGUF_TYPE_UINT64:
            return std::to_string(static_cast<unsigned long long>(gguf_get_val_u64(metadata, key_id)));
        case GGUF_TYPE_INT64:
            return std::to_string(static_cast<long long>(gguf_get_val_i64(metadata, key_id)));
        case GGUF_TYPE_FLOAT32: {
            std::ostringstream output;
            output << gguf_get_val_f32(metadata, key_id);
            return output.str();
        }
        case GGUF_TYPE_FLOAT64: {
            std::ostringstream output;
            output << gguf_get_val_f64(metadata, key_id);
            return output.str();
        }
        case GGUF_TYPE_BOOL:
            return bool_to_string(gguf_get_val_bool(metadata, key_id));
        case GGUF_TYPE_ARRAY: {
            std::ostringstream output;
            output << "array[" << gguf_type_name(gguf_get_arr_type(metadata, key_id))
                   << "," << gguf_get_arr_n(metadata, key_id) << "]";
            return output.str();
        }
        default:
            return std::string("<") + gguf_type_name(value_type) + ">";
    }
}

static void append_gguf_key_if_present(const gguf_context *metadata, const std::string &key) {
    const auto key_id = gguf_find_key(metadata, key.c_str());
    if (key_id < 0) {
        return;
    }

    append_native_diagnostics_line("kv." + key, gguf_value_to_string(metadata, key_id));
}

static void append_gguf_metadata_preflight(const std::string &model_path) {
    append_native_diagnostics_section("gguf_metadata_preflight");
    gguf_init_params metadata_params = {
            /*no_alloc =*/ true,
            /*ctx      =*/ nullptr
    };

    errno = 0;
    gguf_context *metadata = gguf_init_from_file(model_path.c_str(), metadata_params);
    const int metadata_errno = errno;
    if (metadata == nullptr) {
        append_native_diagnostics_line("metadata_load", "failed");
        if (metadata_errno != 0) {
            append_native_diagnostics_line("metadata_errno", format_errno_value(metadata_errno));
        }
        append_native_diagnostics("gguf_init_from_file returned null during metadata preflight\n");
        return;
    }

    append_native_diagnostics_line("metadata_load", "ok");
    append_native_diagnostics_line("gguf_version", std::to_string(gguf_get_version(metadata)));
    append_native_diagnostics_line("gguf_alignment", std::to_string(gguf_get_alignment(metadata)));
    append_native_diagnostics_line("gguf_data_offset", std::to_string(gguf_get_data_offset(metadata)));
    append_native_diagnostics_line("gguf_tensor_count", std::to_string(gguf_get_n_tensors(metadata)));
    append_native_diagnostics_line("gguf_kv_count", std::to_string(gguf_get_n_kv(metadata)));

    std::string architecture = "llama";
    const auto architecture_key_id = gguf_find_key(metadata, "general.architecture");
    if (architecture_key_id >= 0) {
        architecture = gguf_value_to_string(metadata, architecture_key_id);
    }

    append_gguf_key_if_present(metadata, "general.architecture");
    append_gguf_key_if_present(metadata, "general.name");
    append_gguf_key_if_present(metadata, "general.basename");
    append_gguf_key_if_present(metadata, "general.size_label");
    append_gguf_key_if_present(metadata, "general.file_type");
    append_gguf_key_if_present(metadata, "general.quantization_version");
    append_gguf_key_if_present(metadata, "tokenizer.ggml.model");
    append_gguf_key_if_present(metadata, architecture + ".context_length");
    append_gguf_key_if_present(metadata, architecture + ".embedding_length");
    append_gguf_key_if_present(metadata, architecture + ".block_count");
    append_gguf_key_if_present(metadata, architecture + ".feed_forward_length");

    if (gguf_get_n_tensors(metadata) > 0) {
        append_native_diagnostics_line("first_tensor_name", gguf_get_tensor_name(metadata, 0));
        append_native_diagnostics_line(
                "first_tensor_type",
                ggml_type_name(gguf_get_tensor_type(metadata, 0)));
        append_native_diagnostics_line(
                "first_tensor_size_bytes",
                std::to_string(gguf_get_tensor_size(metadata, 0)));
    }

    gguf_free(metadata);
}

struct aichat_load_progress_state {
    float last_progress = -1.0f;
};

static bool aichat_load_progress_callback(const float progress, void *user_data) {
    auto *state = static_cast<aichat_load_progress_state *>(user_data);
    if (state == nullptr ||
        state->last_progress < 0.0f ||
        progress >= 1.0f ||
        progress - state->last_progress >= 0.05f) {
        std::ostringstream output;
        output << std::fixed << std::setprecision(1) << (progress * 100.0f) << "%";
        append_native_diagnostics_line("load_progress", output.str());
        if (state != nullptr) {
            state->last_progress = progress;
        }
    }
    return true;
}

static void clear_native_diagnostics() {
    std::lock_guard<std::mutex> lock(g_diagnostics_mutex);
    g_native_diagnostics.clear();
}

static std::string native_diagnostics_snapshot() {
    std::lock_guard<std::mutex> lock(g_diagnostics_mutex);
    return g_native_diagnostics;
}

static void aichat_llama_log_callback(enum ggml_log_level level,
                                      const char *text,
                                      void *user) {
    append_native_diagnostics(text);
    aichat_android_log_callback(level, text, user);
}

static void release_model_resources() {
    if (g_sampler != nullptr) {
        common_sampler_free(g_sampler);
        g_sampler = nullptr;
    }

    g_chat_templates.reset();

    if (g_batch.token != nullptr || g_batch.embd != nullptr || g_batch.pos != nullptr ||
        g_batch.n_seq_id != nullptr || g_batch.seq_id != nullptr || g_batch.logits != nullptr) {
        llama_batch_free(g_batch);
        g_batch = llama_batch {};
    }

    if (g_context != nullptr) {
        llama_free(g_context);
        g_context = nullptr;
    }

    if (g_model != nullptr) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
}

static int load_model_from_source(
        const std::string &model_label,
        const std::string &load_source_path,
        const std::string &source_kind) {
    llama_model_params model_params = llama_model_default_params();
    aichat_load_progress_state progress_state;
    model_params.progress_callback = aichat_load_progress_callback;
    model_params.progress_callback_user_data = &progress_state;
    model_params.use_mmap = false;
    model_params.use_mlock = false;
    clear_native_diagnostics();

    LOGd("%s: Loading model from (%s): \n%s\n", __func__, source_kind.c_str(), model_label.c_str());
    append_native_diagnostics_section("model_load_attempt");
    append_native_diagnostics_line("model_path", model_label);
    append_native_diagnostics_line("load_source", load_source_path);
    append_native_diagnostics_line("load_source_kind", source_kind);
    append_native_diagnostics_line("params.use_mmap", bool_to_string(model_params.use_mmap));
    append_native_diagnostics_line("params.use_direct_io", bool_to_string(model_params.use_direct_io));
    append_native_diagnostics_line("params.use_mlock", bool_to_string(model_params.use_mlock));
    append_native_diagnostics_line("params.check_tensors", bool_to_string(model_params.check_tensors));
    append_native_diagnostics_line("params.no_alloc", bool_to_string(model_params.no_alloc));
    append_native_diagnostics_line("params.vocab_only", bool_to_string(model_params.vocab_only));
    append_native_runtime_snapshot("before_model_load");

    if (ggml_backend_reg_count() == 0) {
        const bool bootstrap_ok = ensure_backends_ready("before_model_load", true);
        append_backend_registry_snapshot("backend_registry_after_bootstrap_before_model_load");
        if (!bootstrap_ok) {
            append_native_diagnostics_section("backend_bootstrap_failure");
            append_native_diagnostics(
                    "No ggml backends are registered. Aborting model load before llama_model_load_from_file.\n");
            return 1;
        }
    }

    if (model_label.rfind("content://", 0) == 0) {
        append_native_diagnostics_line("model_uri", model_label);
    } else {
        append_model_file_preflight(model_label);
        append_gguf_metadata_preflight(model_label);
    }

    errno = 0;
    auto *model = llama_model_load_from_file(load_source_path.c_str(), model_params);
    const int load_errno = errno;
    if (!model) {
        if (load_errno != 0) {
            append_native_diagnostics_line(
                    "errno_after_llama_model_load_from_file",
                    format_errno_value(load_errno));
        }
        if (progress_state.last_progress < 0.0f) {
            append_native_diagnostics_line("load_progress", "callback never invoked");
        }
        append_native_diagnostics("llama_model_load_from_file returned null\n");
        append_native_diagnostics("model_path: " + model_label + "\n");
        append_native_diagnostics("load_source: " + load_source_path + "\n");
        return 1;
    }

    g_model = model;
    return 0;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_resetNativeDiagnostics(JNIEnv *, jobject) {
    clear_native_diagnostics();
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_lastNativeDiagnostics(JNIEnv *env, jobject) {
    const auto diagnostics = trim_copy(native_diagnostics_snapshot());
    if (diagnostics.empty()) {
        return nullptr;
    }

    return env->NewStringUTF(diagnostics.c_str());
}

extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_init(JNIEnv *env, jobject /*unused*/, jstring nativeLibDir) {
    // Set llama log handler to Android
    clear_native_diagnostics();
    llama_log_set(aichat_llama_log_callback, nullptr);

    const auto *path_to_backend = env->GetStringUTFChars(nativeLibDir, 0);
    g_native_lib_dir = path_to_backend;
    LOGi("Initializing backend bootstrap with native library hint %s", path_to_backend);
    env->ReleaseStringUTFChars(nativeLibDir, path_to_backend);

    const bool bootstrap_ok = ensure_backends_ready("jni_init", true);
    if (!bootstrap_ok) {
        append_native_diagnostics_section("backend_bootstrap_failure");
        append_native_diagnostics("JNI init completed without any registered ggml backend\n");
        LOGe("No ggml backend registered during JNI init");
    } else {
        LOGi("Backend bootstrap completed successfully");
    }

    append_native_runtime_snapshot("post_init");
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_load(JNIEnv *env, jobject, jstring jmodel_path) {
    const auto *model_path = env->GetStringUTFChars(jmodel_path, 0);
    const std::string model_path_string = model_path;
    const int result = load_model_from_source(model_path_string, model_path_string, "path");
    env->ReleaseStringUTFChars(jmodel_path, model_path);
    return result;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_loadFromFd(
        JNIEnv *env,
        jobject,
        jint model_fd,
        jstring jmodel_path) {
    const auto *model_path = env->GetStringUTFChars(jmodel_path, 0);
    const std::string model_path_string = model_path;
    env->ReleaseStringUTFChars(jmodel_path, model_path);

    if (model_fd < 0) {
        clear_native_diagnostics();
        append_native_diagnostics_section("model_load_attempt");
        append_native_diagnostics_line("model_path", model_path_string);
        append_native_diagnostics_line("load_source_kind", "fd");
        append_native_diagnostics("Invalid model file descriptor\n");
        return 1;
    }

    const int dupfd = dup(model_fd);
    if (dupfd == -1) {
        clear_native_diagnostics();
        append_native_diagnostics_section("model_load_attempt");
        append_native_diagnostics_line("model_path", model_path_string);
        append_native_diagnostics_line("load_source_kind", "fd");
        append_native_diagnostics_line("dup_error", format_errno_value(errno));
        return 1;
    }

    const std::string fd_path = "/proc/self/fd/" + std::to_string(dupfd);
    const int result = load_model_from_source(model_path_string, fd_path, "fd");
    close(dupfd);
    return result;
}

static llama_context *init_context(llama_model *model, const int n_ctx = DEFAULT_CONTEXT_SIZE) {
    if (!model) {
        append_native_diagnostics("init_context received a null model\n");
        LOGe("%s: model cannot be null", __func__);
        return nullptr;
    }

    // Multi-threading setup
    const int n_threads = std::max(N_THREADS_MIN, std::min(N_THREADS_MAX,
                                                     (int) sysconf(_SC_NPROCESSORS_ONLN) -
                                                     N_THREADS_HEADROOM));
    LOGi("%s: Using %d threads", __func__, n_threads);

    // Context parameters setup
    llama_context_params ctx_params = llama_context_default_params();
    const int trained_context_size = llama_model_n_ctx_train(model);
    if (n_ctx > trained_context_size) {
        LOGw("%s: Model was trained with only %d context size! Enforcing %d context size...",
             __func__, trained_context_size, n_ctx);
    }
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = BATCH_SIZE;
    ctx_params.n_ubatch = BATCH_SIZE;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    auto *context = llama_init_from_model(model, ctx_params);
    if (context == nullptr) {
        append_native_diagnostics("llama_init_from_model returned null\n");
        LOGe("%s: llama_init_from_model() returned null", __func__);
    }
    return context;
}

static common_sampler *new_sampler(float temp) {
    common_params_sampling sparams;
    sparams.temp = temp;
    return common_sampler_init(g_model, sparams);
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_prepare(
        JNIEnv * /*env*/,
        jobject /*unused*/,
        jint context_size,
        jfloat temperature
) {
    g_context_size = context_size > 0 ? context_size : DEFAULT_CONTEXT_SIZE;
    g_sampler_temp = temperature > 0 ? temperature : DEFAULT_SAMPLER_TEMP;
    append_native_diagnostics_section("prepare_model_resources");
    append_native_diagnostics_line("context_size", std::to_string(g_context_size));
    append_native_diagnostics_line("temperature", std::to_string(g_sampler_temp));
    append_native_diagnostics_line(
            "trained_context_size",
            g_model == nullptr ? "<model not loaded>" : std::to_string(llama_model_n_ctx_train(g_model)));

    auto *context = init_context(g_model, g_context_size);
    if (!context) {
        release_model_resources();
        return 1;
    }
    g_context = context;
    g_batch = llama_batch_init(BATCH_SIZE, 0, 1);
    if (g_batch.token == nullptr && g_batch.embd == nullptr && g_batch.pos == nullptr &&
        g_batch.n_seq_id == nullptr && g_batch.seq_id == nullptr && g_batch.logits == nullptr) {
        append_native_diagnostics("llama_batch_init returned an empty batch\n");
        release_model_resources();
        return 1;
    }
    g_chat_templates = common_chat_templates_init(g_model, "");
    if (!g_chat_templates) {
        append_native_diagnostics("common_chat_templates_init returned null\n");
        release_model_resources();
        return 1;
    }
    g_sampler = new_sampler(g_sampler_temp);
    if (!g_sampler) {
        append_native_diagnostics("common_sampler_init returned null\n");
        release_model_resources();
        return 1;
    }
    return 0;
}

static std::string get_backend() {
    std::vector<std::string> backends;
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto *reg = ggml_backend_reg_get(i);
        std::string name = ggml_backend_reg_name(reg);
        if (name != "CPU") {
            backends.push_back(ggml_backend_reg_name(reg));
        }
    }
    return backends.empty() ? "CPU" : join(backends, ",");
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_systemInfo(JNIEnv *env, jobject /*unused*/) {
    return env->NewStringUTF(llama_print_system_info());
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_benchModel(JNIEnv *env, jobject /*unused*/, jint pp, jint tg,
                                                      jint pl, jint nr) {
    auto *context = init_context(g_model, pp);
    if (!context) {
        const auto *const err_msg = "Fail to init_context! Bench aborted.";
        LOGe(err_msg);
        return env->NewStringUTF(err_msg);
    }

    auto pp_avg = 0.0;
    auto tg_avg = 0.0;
    auto pp_std = 0.0;
    auto tg_std = 0.0;

    const uint32_t n_ctx = llama_n_ctx(context);
    LOGi("n_ctx = %d", n_ctx);

    int i, j;
    int nri;
    for (nri = 0; nri < nr; nri++) {
        LOGi("Benchmark prompt processing (pp = %d)", pp);

        common_batch_clear(g_batch);

        const int n_tokens = pp;
        for (i = 0; i < n_tokens; i++) {
            common_batch_add(g_batch, 0, i, {0}, false);
        }

        g_batch.logits[g_batch.n_tokens - 1] = true;
        llama_memory_clear(llama_get_memory(context), false);

        const auto t_pp_start = ggml_time_us();
        if (llama_decode(context, g_batch) != 0) {
            LOGe("llama_decode() failed during prompt processing");
        }
        const auto t_pp_end = ggml_time_us();

        // bench text generation

        LOGi("Benchmark text generation (tg = %d)", tg);

        llama_memory_clear(llama_get_memory(context), false);
        const auto t_tg_start = ggml_time_us();
        for (i = 0; i < tg; i++) {
            common_batch_clear(g_batch);
            for (j = 0; j < pl; j++) {
                common_batch_add(g_batch, 0, i, {j}, true);
            }

            if (llama_decode(context, g_batch) != 0) {
                LOGe("llama_decode() failed during text generation");
            }
        }
        const auto t_tg_end = ggml_time_us();

        llama_memory_clear(llama_get_memory(context), false);

        const auto t_pp = double(t_pp_end - t_pp_start) / 1000000.0;
        const auto t_tg = double(t_tg_end - t_tg_start) / 1000000.0;

        const auto speed_pp = double(pp) / t_pp;
        const auto speed_tg = double(pl * tg) / t_tg;

        pp_avg += speed_pp;
        tg_avg += speed_tg;

        pp_std += speed_pp * speed_pp;
        tg_std += speed_tg * speed_tg;

        LOGi("pp %f t/s, tg %f t/s", speed_pp, speed_tg);
    }

    llama_free(context);

    pp_avg /= double(nr);
    tg_avg /= double(nr);

    if (nr > 1) {
        pp_std = sqrt(pp_std / double(nr - 1) - pp_avg * pp_avg * double(nr) / double(nr - 1));
        tg_std = sqrt(tg_std / double(nr - 1) - tg_avg * tg_avg * double(nr) / double(nr - 1));
    } else {
        pp_std = 0;
        tg_std = 0;
    }

    char model_desc[128];
    llama_model_desc(g_model, model_desc, sizeof(model_desc));

    const auto model_size = double(llama_model_size(g_model)) / 1024.0 / 1024.0 / 1024.0;
    const auto model_n_params = double(llama_model_n_params(g_model)) / 1e9;

    const auto backend = get_backend();
    std::stringstream result;
    result << std::setprecision(3);
    result << "| model | size | params | backend | test | t/s |\n";
    result << "| --- | --- | --- | --- | --- | --- |\n";
    result << "| " << model_desc << " | " << model_size << "GiB | " << model_n_params << "B | "
           << backend << " | pp " << pp << " | " << pp_avg << " ± " << pp_std << " |\n";
    result << "| " << model_desc << " | " << model_size << "GiB | " << model_n_params << "B | "
           << backend << " | tg " << tg << " | " << tg_avg << " ± " << tg_std << " |\n";
    return env->NewStringUTF(result.str().c_str());
}


/**
 * Completion loop's long-term states:
 * - chat management
 * - position tracking
 */
constexpr const char *ROLE_SYSTEM       = "system";
constexpr const char *ROLE_USER         = "user";
constexpr const char *ROLE_ASSISTANT    = "assistant";

static std::vector<common_chat_msg> chat_msgs;
static llama_pos system_prompt_position;
static llama_pos current_position;

static void reset_long_term_states(const bool clear_kv_cache = true) {
    chat_msgs.clear();
    system_prompt_position = 0;
    current_position = 0;

    if (clear_kv_cache)
        llama_memory_clear(llama_get_memory(g_context), false);
}

/**
 * TODO-hyin: implement sliding-window version as a better alternative
 *
 * Context shifting by discarding the older half of the tokens appended after system prompt:
 * - take the [system_prompt_position] first tokens from the original prompt
 * - take half of the last (system_prompt_position - system_prompt_position) tokens
 * - recompute the logits in batches
 */
static void shift_context() {
    const int n_discard = (current_position - system_prompt_position) / 2;
    LOGi("%s: Discarding %d tokens", __func__, n_discard);
    llama_memory_seq_rm(llama_get_memory(g_context), 0, system_prompt_position, system_prompt_position + n_discard);
    llama_memory_seq_add(llama_get_memory(g_context), 0, system_prompt_position + n_discard, current_position, -n_discard);
    current_position -= n_discard;
    LOGi("%s: Context shifting done! Current position: %d", __func__, current_position);
}

static std::string chat_add_and_format(const std::string &role, const std::string &content) {
    common_chat_msg new_msg;
    new_msg.role = role;
    new_msg.content = content;
    auto formatted = common_chat_format_single(
            g_chat_templates.get(), chat_msgs, new_msg, role == ROLE_USER, /* use_jinja */ false);
    chat_msgs.push_back(new_msg);
    LOGi("%s: Formatted and added %s message: \n%s\n", __func__, role.c_str(), formatted.c_str());
    return formatted;
}

/**
 * Completion loop's short-term states:
 * - stop generation position
 * - token chars caching
 * - current assistant message being generated
 */
static llama_pos stop_generation_position;
static std::string cached_token_chars;
static std::ostringstream assistant_ss;

static void reset_short_term_states() {
    stop_generation_position = 0;
    cached_token_chars.clear();
    assistant_ss.str("");
}

static int decode_tokens_in_batches(
        llama_context *context,
        llama_batch &batch,
        const llama_tokens &tokens,
        const llama_pos start_pos,
        const bool compute_last_logit = false) {
    // Process tokens in batches using the global batch
    LOGd("%s: Decode %d tokens starting at position %d", __func__, (int) tokens.size(), start_pos);
    for (int i = 0; i < (int) tokens.size(); i += BATCH_SIZE) {
        const int cur_batch_size = std::min((int) tokens.size() - i, BATCH_SIZE);
        common_batch_clear(batch);
        LOGv("%s: Preparing a batch size of %d starting at: %d", __func__, cur_batch_size, i);

        // Shift context if current batch cannot fit into the context
        if (start_pos + i + cur_batch_size >= g_context_size - OVERFLOW_HEADROOM) {
            LOGw("%s: Current batch won't fit into context! Shifting...", __func__);
            shift_context();
        }

        // Add tokens to the batch with proper positions
        for (int j = 0; j < cur_batch_size; j++) {
            const llama_token token_id = tokens[i + j];
            const llama_pos position = start_pos + i + j;
            const bool want_logit = compute_last_logit && (i + j == tokens.size() - 1);
            common_batch_add(batch, token_id, position, {0}, want_logit);
        }

        // Decode this batch
        const int decode_result = llama_decode(context, batch);
        if (decode_result) {
            LOGe("%s: llama_decode failed w/ %d", __func__, decode_result);
            return 1;
        }
    }
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_processSystemPrompt(
        JNIEnv *env,
        jobject /*unused*/,
        jstring jsystem_prompt
) {
    // Reset long-term & short-term states
    reset_long_term_states();
    reset_short_term_states();

    // Obtain system prompt from JEnv
    const auto *system_prompt = env->GetStringUTFChars(jsystem_prompt, nullptr);
    LOGd("%s: System prompt received: \n%s", __func__, system_prompt);
    std::string formatted_system_prompt(system_prompt);

    // Format system prompt if applicable
    const bool has_chat_template = common_chat_templates_was_explicit(g_chat_templates.get());
    if (has_chat_template) {
        formatted_system_prompt = chat_add_and_format(ROLE_SYSTEM, system_prompt);
    }
    env->ReleaseStringUTFChars(jsystem_prompt, system_prompt);

    // Tokenize system prompt
    const auto system_tokens = common_tokenize(g_context, formatted_system_prompt,
                                               has_chat_template, has_chat_template);
    for (auto id: system_tokens) {
        LOGv("token: `%s`\t -> `%d`", common_token_to_piece(g_context, id).c_str(), id);
    }

    // Handle context overflow
    const int max_batch_size = g_context_size - OVERFLOW_HEADROOM;
    if ((int) system_tokens.size() > max_batch_size) {
        LOGe("%s: System prompt too long for context! %d tokens, max: %d",
             __func__, (int) system_tokens.size(), max_batch_size);
        return 1;
    }

    // Decode system tokens in batches
    if (decode_tokens_in_batches(g_context, g_batch, system_tokens, current_position)) {
        LOGe("%s: llama_decode() failed!", __func__);
        return 2;
    }

    // Update position
    system_prompt_position = current_position = (int) system_tokens.size();
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_processUserPrompt(
        JNIEnv *env,
        jobject /*unused*/,
        jstring juser_prompt,
        jint n_predict
) {
    // Reset short-term states
    reset_short_term_states();

    // Obtain and tokenize user prompt
    const auto *const user_prompt = env->GetStringUTFChars(juser_prompt, nullptr);
    LOGd("%s: User prompt received: \n%s", __func__, user_prompt);
    std::string formatted_user_prompt(user_prompt);

    // Format user prompt if applicable
    const bool has_chat_template = common_chat_templates_was_explicit(g_chat_templates.get());
    if (has_chat_template) {
        formatted_user_prompt = chat_add_and_format(ROLE_USER, user_prompt);
    }
    env->ReleaseStringUTFChars(juser_prompt, user_prompt);

    // Decode formatted user prompts
    auto user_tokens = common_tokenize(g_context, formatted_user_prompt, has_chat_template, has_chat_template);
    for (auto id: user_tokens) {
        LOGv("token: `%s`\t -> `%d`", common_token_to_piece(g_context, id).c_str(), id);
    }

    // Ensure user prompt doesn't exceed the context size by truncating if necessary.
    const int original_user_prompt_size = (int) user_tokens.size();
    const int max_batch_size = g_context_size - OVERFLOW_HEADROOM;
    if (original_user_prompt_size > max_batch_size) {
        const int skipped_tokens = original_user_prompt_size - max_batch_size;
        user_tokens.resize(max_batch_size);
        LOGw("%s: User prompt too long! Skipped %d tokens!", __func__, skipped_tokens);
    }
    const int effective_user_prompt_size = (int) user_tokens.size();

    // Decode user tokens in batches
    if (decode_tokens_in_batches(g_context, g_batch, user_tokens, current_position, true)) {
        LOGe("%s: llama_decode() failed!", __func__);
        return 2;
    }

    // Update position
    current_position += effective_user_prompt_size;
    stop_generation_position = current_position + n_predict;
    return 0;
}

static bool is_valid_utf8(const char *string) {
    if (!string) { return true; }

    const auto *bytes = (const unsigned char *) string;
    int num;

    while (*bytes != 0x00) {
        if ((*bytes & 0x80) == 0x00) {
            // U+0000 to U+007F
            num = 1;
        } else if ((*bytes & 0xE0) == 0xC0) {
            // U+0080 to U+07FF
            num = 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            // U+0800 to U+FFFF
            num = 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            // U+10000 to U+10FFFF
            num = 4;
        } else {
            return false;
        }

        bytes += 1;
        for (int i = 1; i < num; ++i) {
            if ((*bytes & 0xC0) != 0x80) {
                return false;
            }
            bytes += 1;
        }
    }
    return true;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_generateNextToken(
        JNIEnv *env,
        jobject /*unused*/
) {
    // Infinite text generation via context shifting
    if (current_position >= g_context_size - OVERFLOW_HEADROOM) {
        LOGw("%s: Context full! Shifting...", __func__);
        shift_context();
    }

    // Stop if reaching the marked position
    if (current_position >= stop_generation_position) {
        LOGw("%s: STOP: hitting stop position: %d", __func__, stop_generation_position);
        return nullptr;
    }

    // Sample next token
    const auto new_token_id = common_sampler_sample(g_sampler, g_context, -1);
    common_sampler_accept(g_sampler, new_token_id, true);

    // Populate the batch with new token, then decode
    common_batch_clear(g_batch);
    common_batch_add(g_batch, new_token_id, current_position, {0}, true);
    if (llama_decode(g_context, g_batch) != 0) {
        LOGe("%s: llama_decode() failed for generated token", __func__);
        return nullptr;
    }

    // Update position
    current_position++;

    // Stop if next token is EOG
    if (llama_vocab_is_eog(llama_model_get_vocab(g_model), new_token_id)) {
        LOGd("id: %d,\tIS EOG!\nSTOP.", new_token_id);
        chat_add_and_format(ROLE_ASSISTANT, assistant_ss.str());
        return nullptr;
    }

    // If not EOG, convert to text
    auto new_token_chars = common_token_to_piece(g_context, new_token_id);
    cached_token_chars += new_token_chars;

    // Create and return a valid UTF-8 Java string
    jstring result = nullptr;
    if (is_valid_utf8(cached_token_chars.c_str())) {
        result = env->NewStringUTF(cached_token_chars.c_str());
        LOGv("id: %d,\tcached: `%s`,\tnew: `%s`", new_token_id, cached_token_chars.c_str(), new_token_chars.c_str());

        assistant_ss << cached_token_chars;
        cached_token_chars.clear();
    } else {
        LOGv("id: %d,\tappend to cache", new_token_id);
        result = env->NewStringUTF("");
    }
    return result;
}


extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_unload(JNIEnv * /*unused*/, jobject /*unused*/) {
    // Reset long-term & short-term states
    reset_long_term_states();
    reset_short_term_states();

    // Free up resources
    release_model_resources();
}

extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_shutdown(JNIEnv *, jobject /*unused*/) {
    g_llama_backend_initialized = false;
    llama_backend_free();
}
