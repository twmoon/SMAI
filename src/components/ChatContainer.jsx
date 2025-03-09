import React, { useState, useEffect, useRef } from 'react';
import { useMediaQuery } from 'react-responsive';
import Header from './Header';
import MessageList from './MessageList';
import InputArea from './InputArea';
import messageHandler from '../utils/messageHandler';
import dummyData from '../utils/dummyData';
import '../styles/chatbot.css';

// 메뉴 아이콘 이미지 가져오기
import menuIcon1 from '../assets/media/menuico01.png';  // 학사일정
import menuIcon2 from '../assets/media/menuico02.png';  // 장학
import menuIcon4 from '../assets/media/menuico04.png';  // 식단
import menuIcon5 from '../assets/media/menuico05.png';  // 졸업
import menuIcon6 from '../assets/media/menuico06.png';  // 수업/수강
import menuIcon7 from '../assets/media/menuico07.png';  // 교내연락처
import menuIcon8 from '../assets/media/menuico08.png';  // 캠퍼스맵
import menuIcon11 from '../assets/media/menuico11.png'; // 도서관
import menuIcon12 from '../assets/media/menuico12.png'; // 증명서
import menuIcon15 from '../assets/media/menuico15.png'; // 블랙보드
import menuIcon17 from '../assets/media/menuico17.png'; // FAQ/챗봇소개
import menuIcon18 from '../assets/media/menuico18.png'; // 등록금
import menuIcon19 from '../assets/media/menuico19.png'; // 성적
import menuIcon20 from '../assets/media/menuico20.png'; // 학적
import menuIcon21 from '../assets/media/menuico21.png'; // 셔틀버스
import menuIcon22 from '../assets/media/menuico22.png'; // 공지사항
import menuIcon23 from '../assets/media/menuico23.png'; // 시설
import menuIcon24 from '../assets/media/menuico24.png'; // 학생교류
import menuIcon25 from '../assets/media/menuico25.png'; // 학생상담
import menuIcon26 from '../assets/media/menuico26.png'; // 학생활동
import menuIcon27 from '../assets/media/menuico27.png'; // 학교소개
import menuIcon28 from '../assets/media/menuico28.png'; // 기숙사
import menuIcon29 from '../assets/media/menuico29.png'; // 취업
import menuIcon30 from '../assets/media/menuico30.png'; // 병무
import menuIcon31 from '../assets/media/menuico31.png'; // 규정안내

/**
 * 챗봇 컨테이너 컴포넌트
 */
const ChatContainer = () => {
  console.log('ChatContainer 렌더링 시작');
  
  // 반응형 디자인을 위한 미디어 쿼리 설정
  const isDesktop = useMediaQuery({ query: '(min-width: 1024px)' });
  console.log('현재 화면 환경:', isDesktop ? '데스크톱' : '모바일');
  
  // 컴포넌트 상태 디버깅을 위한 로깅 추가
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState(0);
  const [debugInfo, setDebugInfo] = useState({
    componentsLoaded: false,
    initialMessageSent: false,
    errors: [],
    deviceType: isDesktop ? 'desktop' : 'mobile'
  });
  
  // 메뉴 버튼 데이터
  const menuItems = [
    { icon: menuIcon31, label: '규정안내', action: '규정안내' },
    { icon: menuIcon17, label: 'FAQ/챗봇소개', action: 'FAQ/챗봇소개' },
    { icon: menuIcon7, label: '교내연락처', action: '교내연락처' },
    { icon: menuIcon4, label: '식단', action: '식단' },
    { icon: menuIcon2, label: '장학', action: '장학' },
    { icon: menuIcon18, label: '등록금', action: '등록금' },
    { icon: menuIcon1, label: '학사일정', action: '학사일정' },
    { icon: menuIcon12, label: '증명서', action: '증명서' },
    { icon: menuIcon6, label: '수업/수강', action: '수업/수강' },
    { icon: menuIcon19, label: '성적', action: '성적' },
    { icon: menuIcon20, label: '학적', action: '학적' },
    { icon: menuIcon8, label: '캠퍼스맵', action: '캠퍼스맵' },
    { icon: menuIcon21, label: '셔틀버스', action: '셔틀버스' },
    { icon: menuIcon22, label: '공지사항', action: '공지사항' },
    { icon: menuIcon5, label: '졸업', action: '졸업' },
    { icon: menuIcon11, label: '도서관', action: '도서관' },
    { icon: menuIcon23, label: '시설', action: '시설' },
    { icon: menuIcon15, label: '블랙보드', action: '블랙보드' },
    { icon: menuIcon24, label: '학생교류', action: '학생교류' },
    { icon: menuIcon25, label: '학생상담', action: '학생상담' },
    { icon: menuIcon26, label: '학생활동', action: '학생활동' },
    { icon: menuIcon27, label: '학교소개', action: '학교소개' },
    { icon: menuIcon28, label: '기숙사', action: '기숙사' },
    { icon: menuIcon29, label: '취업', action: '취업' },
    { icon: menuIcon30, label: '병무', action: '병무' },
  ];
  
  // 페이지당 버튼 수
  const buttonsPerPage = isDesktop ? 16 : 8;
  
  // 총 페이지 수 계산
  const totalPages = Math.ceil(menuItems.length / buttonsPerPage);
  
  // 현재 페이지의 버튼만 표시
  const getCurrentPageButtons = () => {
    const startIdx = currentPage * buttonsPerPage;
    const endIdx = startIdx + buttonsPerPage;
    return menuItems.slice(startIdx, endIdx);
  };
  
  // 페이지 변경 핸들러
  const changePage = (pageNum) => {
    setCurrentPage(pageNum);
  };
  
  // 스와이프 관련 상태 추가
  const [touchStart, setTouchStart] = useState(null);
  const [touchEnd, setTouchEnd] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [startX, setStartX] = useState(0);
  const menuGridRef = useRef(null);

  // 스와이프 거리 최소값 설정
  const minSwipeDistance = 50;

  // 터치 핸들러 (모바일)
  const onTouchStart = (e) => {
    setTouchEnd(null);
    setTouchStart(e.targetTouches[0].clientX);
  };

  const onTouchMove = (e) => {
    setTouchEnd(e.targetTouches[0].clientX);
  };

  const onTouchEnd = () => {
    if (!touchStart || !touchEnd) return;
    
    const distance = touchStart - touchEnd;
    const isLeftSwipe = distance > minSwipeDistance;
    const isRightSwipe = distance < -minSwipeDistance;
    
    if (isLeftSwipe && currentPage < totalPages - 1) {
      // 왼쪽으로 스와이프: 다음 페이지
      changePage(currentPage + 1);
    } else if (isRightSwipe && currentPage > 0) {
      // 오른쪽으로 스와이프: 이전 페이지
      changePage(currentPage - 1);
    }
  };

  // 마우스 핸들러 (데스크톱)
  const onMouseDown = (e) => {
    setIsDragging(true);
    setStartX(e.clientX);
    
    // 드래그 중 텍스트 선택 방지
    e.preventDefault();
  };

  const onMouseMove = (e) => {
    if (!isDragging) return;
    
    // 드래그 중에 텍스트 선택 방지
    window.getSelection().removeAllRanges();
    
    const currentX = e.clientX;
    const diff = startX - currentX;
    
    // 드래그 시 시각적 피드백 (선택사항)
    if (menuGridRef.current) {
      const scrollAmount = diff * 0.5; // 드래그 속도 조절
      menuGridRef.current.scrollLeft += scrollAmount;
    }
  };

  const onMouseUp = (e) => {
    if (!isDragging) return;
    
    const endX = e.clientX;
    const distance = startX - endX;
    
    // 드래그 거리가 충분하면 페이지 전환
    if (Math.abs(distance) > minSwipeDistance) {
      if (distance > 0 && currentPage < totalPages - 1) {
        // 왼쪽으로 드래그: 다음 페이지
        changePage(currentPage + 1);
      } else if (distance < 0 && currentPage > 0) {
        // 오른쪽으로 드래그: 이전 페이지
        changePage(currentPage - 1);
      }
    }
    
    setIsDragging(false);
  };

  // 드래그 중 마우스가 요소 밖으로 나갔을 때 처리
  const onMouseLeave = () => {
    if (isDragging) {
      setIsDragging(false);
    }
  };

  // 컴포넌트 마운트/언마운트 시 전역 이벤트 설정/해제
  useEffect(() => {
    const handleGlobalMouseUp = () => {
      if (isDragging) {
        setIsDragging(false);
      }
    };

    // 전역 mouseup 이벤트 리스너 추가 (드래그 중 요소 밖에서 마우스를 놓는 경우 처리)
    window.addEventListener('mouseup', handleGlobalMouseUp);

    return () => {
      window.removeEventListener('mouseup', handleGlobalMouseUp);
    };
  }, [isDragging]);
  
  // 현재 시간 구하기
  const getCurrentTime = () => {
    const now = new Date();
    const hours = now.getHours();
    const minutes = now.getMinutes() < 10 ? '0' + now.getMinutes() : now.getMinutes();
    return `${hours < 10 ? '0' + hours : hours}:${minutes}`;
  };
  
  // 컴포넌트 마운트 시 초기 인사 메시지 표시
  useEffect(() => {
    try {
      console.log('ChatContainer useEffect 실행', dummyData);
      
      if (!dummyData || !dummyData.greeting) {
        throw new Error('dummyData 또는 greeting 객체가 없습니다');
      }
      
      const initialGreeting = messageHandler.createMessage(
        "안녕하세요! 상명대학교 챗봇 새미입니다. 무엇을 도와드릴까요?",
        'bot',
        { 
          type: 'text',
          options: ["입학 정보를 알려주세요", "학사 일정을 알려주세요", "캠퍼스를 안내해주세요", "장학금 관련 정보를 알려주세요"] 
        }
      );
      
      // 현재 시간 추가
      initialGreeting.time = getCurrentTime();
      
      setMessages([initialGreeting]);
      setDebugInfo(prev => ({...prev, initialMessageSent: true}));
      console.log('초기 메시지 설정 완료', initialGreeting);
    } catch (error) {
      console.error('초기 메시지 설정 중 오류 발생:', error);
      setDebugInfo(prev => ({...prev, errors: [...prev.errors, error.message]}));
    }
  }, []);
  
  // 컴포넌트 로드 확인
  useEffect(() => {
    setDebugInfo(prev => ({...prev, componentsLoaded: true}));
    console.log('모든 컴포넌트 로드 완료');
  }, []);

  // 화면 크기 변경 감지
  useEffect(() => {
    setDebugInfo(prev => ({...prev, deviceType: isDesktop ? 'desktop' : 'mobile'}));
    console.log('화면 크기 변경 감지:', isDesktop ? '데스크톱' : '모바일');
  }, [isDesktop]);

  // 사용자 메시지 처리 함수
  const handleSendMessage = (text) => {
    // 사용자 메시지 추가
    const userMessage = messageHandler.createMessage(text, 'user');
    // 현재 시간 추가
    userMessage.time = getCurrentTime();
    setMessages(prevMessages => [...prevMessages, userMessage]);
    
    // 로딩 상태 활성화
    setIsLoading(true);
    
    // 봇 응답 생성 (실제로는 API 호출로 대체될 수 있음)
    setTimeout(() => {
      const response = messageHandler.generateResponse(text);
      const botMessage = messageHandler.createMessage(
        response.content,
        'bot',
        { 
          type: response.type, 
          data: response.data,
          options: response.options
        }
      );
      
      // 현재 시간 추가
      botMessage.time = getCurrentTime();
      
      setMessages(prevMessages => [...prevMessages, botMessage]);
      setIsLoading(false);
    }, 1000); // 1초 후 응답 생성하여 자연스러운 대화 느낌 구현
  };
  
  // 빠른 응답 옵션 클릭 처리
  const handleOptionClick = (option) => {
    handleSendMessage(option);
  };
  
  // 메뉴 버튼 클릭 처리
  const handleMenuButtonClick = (action) => {
    handleSendMessage(action);
  };
  
  return (
    <div className="chatbot-container">
      {/* 디버그 모드일 때만 표시되는 디버깅 정보 패널 */}
      {process.env.NODE_ENV === 'development' && (
        <div style={{padding: '10px', backgroundColor: '#e3f2fd', fontSize: '12px', display: 'none'}}>
          <h4>디버그 정보</h4>
          <pre>{JSON.stringify(debugInfo, null, 2)}</pre>
          <button onClick={() => console.log('현재 상태:', {messages, isLoading, debugInfo})}>
            콘솔에 상태 출력
          </button>
        </div>
      )}
      
      <Header deviceType={isDesktop ? 'desktop' : 'mobile'} />
      
      <div className="chat-content-wrapper">
        <MessageList 
          messages={messages} 
          onOptionClick={handleOptionClick}
          deviceType={isDesktop ? 'desktop' : 'mobile'} 
        />
        
        {isLoading && (
          <div className="loading-indicator">
            <div className="loading-dots">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
        
        <InputArea 
          onSendMessage={handleSendMessage} 
          deviceType={isDesktop ? 'desktop' : 'mobile'} 
        />
      </div>
      
      {/* 메뉴 버튼 그리드 */}
      <div 
        ref={menuGridRef}
        className={`quick-menu-grid ${totalPages > 1 ? 'has-multiple-pages' : ''}`}
        onTouchStart={onTouchStart}
        onTouchMove={onTouchMove}
        onTouchEnd={onTouchEnd}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseLeave}
        style={{ position: 'relative' }}
      >
        {getCurrentPageButtons().map((item, index) => (
          <button 
            key={index} 
            className="menu-button"
            onClick={() => handleMenuButtonClick(item.action)}
            onMouseDown={(e) => e.stopPropagation()} // 버튼 클릭 시 드래그 시작 방지
          >
            <img 
              src={item.icon} 
              alt={item.label} 
            />
            <span>{item.label}</span>
          </button>
        ))}
      </div>
      
      {/* 페이지네이션 도트 */}
      {totalPages > 1 && (
        <div className="pagination-dots">
          {Array.from({ length: totalPages }).map((_, index) => (
            <div 
              key={index}
              className={`dot ${currentPage === index ? 'active' : ''}`}
              onClick={() => changePage(index)}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default ChatContainer;
