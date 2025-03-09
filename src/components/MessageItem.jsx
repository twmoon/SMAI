import React from 'react';
import '../styles/chatbot.css';
import chatbotIcon from '../assets/img/ico_chatbot.png';

/**
 * 개별 메시지 아이템 컴포넌트
 * @param {object} props
 * @param {object} props.message - 메시지 객체
 * @param {function} props.onOptionClick - 옵션 클릭 핸들러
 * @param {string} props.deviceType - 'desktop' 또는 'mobile'
 */
const MessageItem = ({ message, onOptionClick, deviceType = 'mobile' }) => {
  const { text, sender, time, type, data, options } = message;
  const isUser = sender === 'user';
  const containerClass = isUser 
    ? `user-message-container ${deviceType === 'desktop' ? 'desktop-user-message-container' : ''}` 
    : `bot-message-container ${deviceType === 'desktop' ? 'desktop-bot-message-container' : ''}`;
  const messageClass = isUser 
    ? `message-item user-message ${deviceType === 'desktop' ? 'desktop-user-message' : ''}` 
    : `message-item bot-message ${deviceType === 'desktop' ? 'desktop-bot-message' : ''}`;

  // 일반 텍스트 메시지 렌더링
  const renderTextMessage = () => (
    <div className={messageClass}>
      {text}
    </div>
  );

  // 학사 일정 데이터 렌더링
  const renderCalendarData = () => (
    <div className={messageClass}>
      <p>{text}</p>
      <ul className={`calendar-list ${deviceType === 'desktop' ? 'desktop-calendar-list' : ''}`}>
        {data.map((item, index) => (
          <li key={index} className="calendar-item">
            <span className="calendar-title">{item.title}</span>
            <span className="calendar-date">{item.date}</span>
          </li>
        ))}
      </ul>
    </div>
  );

  // 장학금 정보 렌더링
  const renderScholarshipData = () => (
    <div className={messageClass}>
      <p>{text}</p>
      <div className={`scholarship-list ${deviceType === 'desktop' ? 'desktop-scholarship-list' : ''}`}>
        {data.map((item, index) => (
          <div key={index} className="scholarship-item">
            <h4>{item.name}</h4>
            <p><strong>대상:</strong> {item.eligibility}</p>
            <p><strong>금액:</strong> {item.amount}</p>
            <p><strong>신청방법:</strong> {item.application}</p>
          </div>
        ))}
      </div>
    </div>
  );

  // 시설 정보 렌더링
  const renderFacilityData = () => (
    <div className={messageClass}>
      <p>{text}</p>
      <div className={`facility-list ${deviceType === 'desktop' ? 'desktop-facility-list' : ''}`}>
        {data.map((item, index) => (
          <div key={index} className="facility-item">
            <h4>{item.name}</h4>
            <p><strong>위치:</strong> {item.location}</p>
            <p><strong>운영시간:</strong> {item.hours}</p>
            <p>{item.description}</p>
          </div>
        ))}
      </div>
    </div>
  );

  // 빠른 응답 옵션 렌더링
  const renderOptions = () => (
    options && (
      <div className="quick-replies">
        {options.map((option, index) => (
          <button 
            key={index} 
            className="quick-reply-button"
            onClick={() => onOptionClick && onOptionClick(option)}
          >
            {option}
          </button>
        ))}
      </div>
    )
  );

  // 메시지 타입에 따라 렌더링 방식 결정
  const renderMessage = () => {
    switch (type) {
      case 'calendar':
        return renderCalendarData();
      case 'scholarships':
        return renderScholarshipData();
      case 'facilities':
        return renderFacilityData();
      default:
        return renderTextMessage();
    }
  };

  return (
    <div className={containerClass} style={{ marginBottom: '20px' }}>
      {!isUser && (
        <div style={{ display: 'flex', flexDirection: 'column', width: '100%' }}>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
            <img 
              src={chatbotIcon} 
              alt="Chatbot" 
              className="message-avatar"
              style={{ 
                width: '45px', 
                height: '45px', 
                marginRight: '12px', 
                marginTop: '0px',
                objectFit: 'contain'
              }} 
            />
            <div style={{ 
              display: 'flex', 
              alignItems: 'center',
              fontWeight: '600',
              fontSize: '22px',
              color: '#222',
              letterSpacing: '-0.5px',
              marginLeft: '2px',
              fontFamily: 'Arial, sans-serif'
            }}>
              새미(SMAI)
            </div>
          </div>
          
          <div style={{ 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'flex-start',
            paddingLeft: '57px', /* chatbot icon width + margin */
            flexGrow: 1
          }}>
            {renderMessage()}
            <div className="message-time">{time}</div>
            {options && options.length > 0 && renderOptions()}
          </div>
        </div>
      )}
      
      {isUser && (
        <div style={{ 
          display: 'flex', 
          flexDirection: 'column', 
          alignItems: 'flex-end',
          width: '100%'
        }}>
          {renderMessage()}
          <div className="message-time">{time}</div>
        </div>
      )}
    </div>
  );
};

export default MessageItem;
