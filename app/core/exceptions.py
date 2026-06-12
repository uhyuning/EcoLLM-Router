# [역할] 프로젝트 내 커스텀 예외 클래스 정의.
# LLMError                  : 모든 LLM 호출 예외의 기반 클래스 (status_code 포함)
# LLMRateLimitError         : 재시도 횟수를 모두 소진한 뒤에도 429가 지속될 때
# LLMServiceUnavailableError: 재시도 소진 후에도 서비스 불가 상태일 때
# LLMAuthError              : API 키·권한 오류 (재시도해도 해소 불가)
# LLMInvalidResponseError   : 응답 파싱 실패 (콘텐츠 필터 등)
#
# 담당: 하윤


class LLMError(Exception):
    """LLM 호출 관련 모든 예외의 기반 클래스.

    status_code를 예외 객체에 직접 담는 이유:
    FastAPI exception_handler에 기반 클래스 하나만 등록해두면
    새로운 하위 예외를 추가해도 핸들러를 수정하지 않아도 된다.
    핸들러 쪽에 if/elif 매핑 테이블을 두면 예외 추가마다 두 파일을 동시에 고쳐야 한다.
    """

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMRateLimitError(LLMError):
    """Gemini API가 429를 반환하고 재시도를 모두 소진한 경우."""

    def __init__(self, message: str = "Rate limit exceeded. Please retry later.") -> None:
        # 429를 그대로 클라이언트에게 전달 — API 소비자가 자체 백오프를 구현할 수 있도록
        super().__init__(message, status_code=429)


class LLMServiceUnavailableError(LLMError):
    """Gemini 서비스가 일시 불가(5xx)이고 재시도를 모두 소진한 경우."""

    def __init__(self, message: str = "LLM service unavailable. Please retry later.") -> None:
        # 503: "우리 서버" 문제가 아닌 "상류(Gemini)" 문제임을 HTTP 시맨틱으로 구분
        super().__init__(message, status_code=503)


class LLMAuthError(LLMError):
    """API 키 누락·만료·권한 부족(401/403). 재시도해도 해소되지 않는다."""

    def __init__(self, message: str = "LLM authentication failed. Check your API key.") -> None:
        # 원본이 403이더라도 401로 통일 — 클라이언트에게는 "인증 실패"라는 사실만 중요
        super().__init__(message, status_code=401)


class LLMInvalidResponseError(LLMError):
    """LLM이 텍스트를 반환하지 않은 경우 (콘텐츠 필터 차단 등).

    502 Bad Gateway: 우리 서버는 정상이지만 상류(Gemini)에서 사용할 수 없는 응답이 왔음을 나타낸다.
    """

    def __init__(self, message: str = "Invalid or empty response from LLM.") -> None:
        super().__init__(message, status_code=502)
