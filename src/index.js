import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './styles/main.css';

console.log('React 앱 초기화 시작');

try {
  const root = createRoot(document.getElementById('root'));
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
  console.log('React 앱 마운트 성공');
} catch (error) {
  console.error('React 앱 마운트 중 오류 발생:', error);
}
