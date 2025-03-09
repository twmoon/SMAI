// src/App.jsx
import React from 'react';
import ChatContainer from './components/ChatContainer';
import ErrorBoundary from './components/ErrorBoundary';
import './styles/main.css';

const App = () => {
  // 앱 로딩 시 콘솔에 디버깅 정보 출력
  console.log('App 컴포넌트 렌더링 시작');
  
  return (
    <div className="app">
      <ErrorBoundary>
        <React.StrictMode>
          <ChatContainer />
        </React.StrictMode>
      </ErrorBoundary>
    </div>
  );
};

export default App;
