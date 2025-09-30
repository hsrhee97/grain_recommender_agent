## LLM 추출 모드 사용법

규칙 기반 추출 대신 OpenAI 모델을 사용하려면 다음 단계를 수행하세요.

1. 의존성 설치
   ```bash
   pip install -r requirements.txt
   ```
   
2. 사용할 LLM 공급자와 자격 증명을 환경 변수로 지정합니다. 기본값은 OpenAI이며, Gemini도 지원합니다.
   - **OpenAI**
     ```bash
     export LLM_PROVIDER=openai
     export OPENAI_API_KEY="sk-..."
     export OPENAI_MODEL="gpt-4o-mini"  # 필요 시 다른 모델 지정
     ```
   - **Gemini**
     ```bash
     export LLM_PROVIDER=gemini
     export GEMINI_API_KEY="your-gemini-key"
     export GEMINI_MODEL="gemini-2.0-flash"  # 필요 시 다른 모델 지정
     ```

3. LLM 모드를 활성화하여 플로우를 실행합니다. 환경 변수를 사용하거나 함수 인자를 통해 제어할 수 있습니다.
   ```bash
   export GRAIN_AGENT_USE_LLM=true

   python - <<'PY'
   from app.agent.graph import run_agent_flow

   result = run_agent_flow(
       "혈당 관리가 필요하고 주 6번 정도 밥을 먹어요. 찰진 식감이 좋고 보리는 빼주세요.",
       user_id="demo",
       use_llm=True,
   )
   print(result["message"])
   PY
   ```

LLM 호출에 실패하거나 API 설정이 누락된 경우에는 자동으로 기존 규칙 기반 추출로 폴백합니다. 폴백 없이 강제 실패시키고 싶다면 `fallback_to_rules=False`를 전달하세요.

