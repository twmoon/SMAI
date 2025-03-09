/* eslint-disable no-restricted-globals */

// 캐시 이름 설정
const CACHE_NAME = 'smu-chatbot-v1';

// 캐시할 정적 자산 목록
const urlsToCache = [
  '/',
  '/index.html',
  '/static/js/main.js',
  '/static/css/main.css',
  '/assets/img/ico_chatbot.png',
  '/assets/img/greeting.png',
  '/assets/media/dark_ico_menu.svg',
  '/assets/media/dark_ico_refresh.svg',
  '/assets/media/dark_ico_send2.svg',
  '/assets/media/dark_triangle.svg',
  '/assets/media/ico_languages.svg',
  '/assets/fonts/NotoSansKR-Regular.woff2'
];

// 서비스 워커 설치 이벤트
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('Opened cache');
        return cache.addAll(urlsToCache);
      })
  );
});

// 네트워크 요청 가로채기
self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // 캐시에서 찾았으면 캐시된 응답 반환
        if (response) {
          return response;
        }

        // 캐시에 없으면 네트워크 요청
        return fetch(event.request)
          .then(response => {
            // 유효한 응답인지 확인하고 캐시에 복사본 저장
            if (!response || response.status !== 200 || response.type !== 'basic') {
              return response;
            }

            const responseToCache = response.clone();

            caches.open(CACHE_NAME)
              .then(cache => {
                cache.put(event.request, responseToCache);
              });

            return response;
          });
      })
  );
});

// 오래된 캐시 정리
self.addEventListener('activate', event => {
  const cacheWhitelist = [CACHE_NAME];

  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheWhitelist.indexOf(cacheName) === -1) {
            // 현재 화이트리스트에 없는 캐시는 삭제
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});
