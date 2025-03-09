import React, { useState } from 'react';
import '../styles/chatbot.css';

// 전송 버튼 아이콘 임포트
import sendIcon from '../assets/media/dark_ico_send2.svg';

/**
 * 메시지 입력 영역 컴포넌트
 * @param {object} props
 * @param {function} props.onSendMessage - 메시지 전송 핸들러
 * @param {string} props.deviceType - 'desktop' 또는 'mobile'
 */
const InputArea = ({ onSendMessage, deviceType = 'mobile' }) => {
  const [message, setMessage] = useState('');
  
  // 입력 변경 핸들러
  const handleInputChange = (e) => {
    setMessage(e.target.value);
  };
  
  // 메시지 제출 핸들러
  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (message.trim()) {
      onSendMessage(message);
      setMessage('');
    }
  };
  
  // 엔터 키 누름 핸들러
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };
  
  return (
    <form className={`input-area ${deviceType === 'desktop' ? 'desktop-input-area' : ''}`} onSubmit={handleSubmit}>
      <input
        type="text"
        className={`message-input ${deviceType === 'desktop' ? 'desktop-message-input' : ''}`}
        placeholder="메시지를 입력하세요..."
        value={message}
        onChange={handleInputChange}
        onKeyPress={handleKeyPress}
      />
      <button type="submit" className="send-button" aria-label="메시지 전송">
        <img 
          src={sendIcon} 
          alt="Send" 
          style={{ 
            verticalAlign: 'middle',
            display: 'block',
            opacity: '0.6',
            filter: 'brightness(1.2)'
          }}
        />
      </button>
    </form>
  );
};

export default InputArea;
