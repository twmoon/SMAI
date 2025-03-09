import React, { useEffect, useRef } from 'react';
import MessageItem from './MessageItem';
import '../styles/chatbot.css';

/**
 * 메시지 목록 컴포넌트
 * @param {object} props
 * @param {array} props.messages - 메시지 객체 배열
 * @param {function} props.onOptionClick - 옵션 클릭 핸들러
 * @param {string} props.deviceType - 'desktop' 또는 'mobile'
 */
const MessageList = ({ messages, onOptionClick, deviceType = 'mobile' }) => {
  const messagesEndRef = useRef(null);
  
  // 새 메시지가 추가될 때 스크롤 이동
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  return (
    <div className={`message-list ${deviceType === 'desktop' ? 'desktop-message-list' : ''}`}>
      {messages.map((message) => (
        <MessageItem 
          key={message.id} 
          message={message} 
          onOptionClick={onOptionClick}
          deviceType={deviceType}
        />
      ))}
      <div ref={messagesEndRef} />
    </div>
  );
};

export default MessageList;
