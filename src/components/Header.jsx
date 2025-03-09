import React from 'react';
import '../styles/chatbot.css';

// 이미지 임포트
import chatbotLogo from '../assets/img/greeting.png';
import refreshIcon from '../assets/media/dark_ico_refresh.png';
import menuIcon from '../assets/media/dark_ico_menu.svg';

/**
 * 챗봇 헤더 컴포넌트
 * @param {object} props
 * @param {string} props.deviceType - 'desktop' 또는 'mobile'
 */
const Header = ({ deviceType = 'mobile' }) => {
  // 새로고침 핸들러
  const handleRefresh = () => {
    window.location.reload();
  };
  
  return (
    <div className={`chatbot-header ${deviceType === 'desktop' ? 'desktop-header' : ''}`}>
      <div style={{ display: 'flex', alignItems: 'center' }}>
        <img 
          src={chatbotLogo} 
          alt="SMU Chatbot" 
          className="chatbot-logo" 
          style={{
            width: '45px',
            height: '45px',
            objectFit: 'contain',
            marginRight: '12px'
          }}
        />
        <div className="chatbot-title">상명대학교 챗봇 새미</div>
      </div>
      <div className="chatbot-controls">
        <button className="control-button" onClick={handleRefresh}>
          <img 
            src={refreshIcon} 
            alt="새로고침" 
            style={{ width: '24px', height: '24px' }}
          />
        </button>
        <button className="control-button">
          <img 
            src={menuIcon} 
            alt="메뉴" 
          />
        </button>
      </div>
    </div>
  );
};

export default Header;
