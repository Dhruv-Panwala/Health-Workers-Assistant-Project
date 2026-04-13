import React from "react";
import "./Header.css";

function Header() {
  return (
    <header className="header">
      <div className="header-content">
        <div className="logo-container">
          <div className="logo">
            <div className="logo-icon">
              <img src="/Logos/Otermans Institute.png" alt="Otermans Institute" />
            </div>
          </div>

          <div className="logo">
            <div className="logo-icon">
              <img
                src="/Logos/Kenya_medical_training_college.png"
                alt="Kenya_medical_training_college"
                className="Kenya_medical_training_college"
              />
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

export default Header;
