import dummyData from './dummyData';

/**
 * 챗봇 메시지 처리 유틸리티
 */
const messageHandler = {
  /**
   * 사용자 메시지에 대한 응답을 생성합니다.
   * @param {string} userMessage - 사용자 입력 메시지
   * @returns {object} - 봇 응답 객체
   */
  generateResponse(userMessage) {
    // 사용자 메시지를 소문자로 변환하여 키워드 매칭
    const message = userMessage.toLowerCase();
    
    // 인사 키워드 체크
    if (this.containsKeywords(message, ['안녕', '하이', '반가워', '시작'])) {
      return {
        type: 'text',
        content: dummyData.greeting.text,
        options: dummyData.greeting.options
      };
    }
    
    // 입학 정보 관련 키워드 체크
    if (this.containsKeywords(message, ['입학', '지원', '수시', '정시', '전형', '모집'])) {
      return {
        type: 'text',
        content: "상명대학교 입학처 홈페이지(admission.smu.ac.kr)에서 자세한 입학 정보를 확인하실 수 있습니다. 수시모집은 9월, 정시모집은 12월에 원서접수가 시작됩니다. 구체적인 전형 정보를 알고 싶으시면 '수시전형', '정시전형'과 같이 질문해 주세요."
      };
    }
    
    // 학사 일정 관련 키워드 체크
    if (this.containsKeywords(message, ['학사', '일정', '캘린더', '개강', '종강', '시험', '수강'])) {
      return {
        type: 'calendar',
        content: "2025학년도 주요 학사 일정입니다.",
        data: dummyData.academicCalendar
      };
    }
    
    // 장학금 관련 키워드 체크
    if (this.containsKeywords(message, ['장학', '장학금', '학비', '등록금', '지원금'])) {
      return {
        type: 'scholarships',
        content: "상명대학교에서 제공하는 주요 장학금 정보입니다.",
        data: dummyData.scholarships
      };
    }
    
    // 시설 관련 키워드 체크
    if (this.containsKeywords(message, ['시설', '캠퍼스', '도서관', '학식', '식당', '기숙사', '위치'])) {
      return {
        type: 'facilities',
        content: "상명대학교 주요 시설 안내입니다.",
        data: dummyData.facilities
      };
    }
    
    // FAQ 확인
    const faqMatch = this.findFaqMatch(message);
    if (faqMatch) {
      return {
        type: 'text',
        content: faqMatch.answer
      };
    }
    
    // 응답을 찾지 못한 경우 Fallback 응답 반환
    return {
      type: 'text',
      content: this.getRandomFallbackResponse()
    };
  },
  
  /**
   * 메시지에 특정 키워드가 포함되어 있는지 확인합니다.
   * @param {string} message - 검사할 메시지
   * @param {Array} keywords - 검색할 키워드 배열
   * @returns {boolean} - 키워드 포함 여부
   */
  containsKeywords(message, keywords) {
    return keywords.some(keyword => message.includes(keyword));
  },
  
  /**
   * 메시지와 일치하는 FAQ를 찾습니다.
   * @param {string} message - 사용자 메시지
   * @returns {object|null} - 일치하는 FAQ 객체 또는 null
   */
  findFaqMatch(message) {
    return dummyData.faq.find(faq => {
      const questionLower = faq.question.toLowerCase();
      return this.calculateSimilarity(message, questionLower) > 0.6;
    });
  },
  
  /**
   * 두 문자열 간의 유사성을 계산합니다(간단한 구현).
   * @param {string} str1 - 첫 번째 문자열
   * @param {string} str2 - 두 번째 문자열
   * @returns {number} - 유사성 점수 (0~1)
   */
  calculateSimilarity(str1, str2) {
    const words1 = str1.split(' ');
    const words2 = str2.split(' ');
    
    let matchCount = 0;
    words1.forEach(word => {
      if (words2.includes(word)) {
        matchCount++;
      }
    });
    
    // 최소 단어 수로 나누어 유사성 점수 계산
    const minWordCount = Math.min(words1.length, words2.length);
    return minWordCount === 0 ? 0 : matchCount / minWordCount;
  },
  
  /**
   * 랜덤한 Fallback 응답을 반환합니다.
   * @returns {string} - 랜덤 Fallback 응답
   */
  getRandomFallbackResponse() {
    const fallbacks = dummyData.fallbackResponses;
    const randomIndex = Math.floor(Math.random() * fallbacks.length);
    return fallbacks[randomIndex];
  },
  
  /**
   * 현재 시간을 HH:MM 형식으로 반환합니다.
   * @returns {string} - 현재 시간 문자열
   */
  getCurrentTime() {
    const now = new Date();
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    return `${hours}:${minutes}`;
  },
  
  /**
   * 새 메시지 객체를 생성합니다.
   * @param {string} text - 메시지 텍스트
   * @param {string} sender - 발신자 ('user' 또는 'bot')
   * @param {object} additionalData - 추가 데이터
   * @returns {object} - 메시지 객체
   */
  createMessage(text, sender, additionalData = {}) {
    return {
      id: Date.now().toString(),
      text: text,
      sender: sender,
      time: this.getCurrentTime(),
      ...additionalData
    };
  }
};

export default messageHandler;
