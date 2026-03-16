#include <jni.h>
#include <string>

extern "C" JNIEXPORT jlong JNICALL
Java_com_healthworker_assistant_LlmService_initModel(
        JNIEnv *env, jobject /* thiz */, jstring modelPath_) {
    const char *modelPath = env->GetStringUTFChars(modelPath_, nullptr);
    std::string path(modelPath == nullptr ? "" : modelPath);
    env->ReleaseStringUTFChars(modelPath_, modelPath);

    if (path.empty()) {
        return 0;
    }

    // Placeholder handle until full llama.cpp token generation is wired in JNI.
    return static_cast<jlong>(1);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_healthworker_assistant_LlmService_inferNative(
        JNIEnv *env, jobject /* thiz */, jlong modelPtr, jstring prompt_) {
    if (modelPtr == 0) {
        return env->NewStringUTF("");
    }

    const char *prompt = env->GetStringUTFChars(prompt_, nullptr);
    std::string promptText(prompt == nullptr ? "" : prompt);
    env->ReleaseStringUTFChars(prompt_, prompt);

    if (promptText.empty()) {
        return env->NewStringUTF("");
    }

    return env->NewStringUTF("");
}

extern "C" JNIEXPORT void JNICALL
Java_com_healthworker_assistant_LlmService_releaseModel(
        JNIEnv * /* env */, jobject /* thiz */, jlong /* modelPtr */) {
}
