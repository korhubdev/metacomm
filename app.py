import os
import json
import re
from typing import List, Dict, Any, Optional

import streamlit as st
from openai import OpenAI

# =========================
# Config
# =========================
st.set_page_config(page_title="금융소비자보호법 제21조 위반 점검(프로토타입)", layout="wide")

# Secrets / Env / Default 우선순위로 모델/키를 가져오도록 구성
def get_openai_api_key() -> str:
    # 1) secrets(openai 섹션) -> 2) secrets 루트 -> 3) env
    if "openai" in st.secrets and "OPENAI_API_KEY" in st.secrets["openai"]:
        return st.secrets["openai"]["OPENAI_API_KEY"]
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    return os.getenv("OPENAI_API_KEY", "")

def get_openai_model() -> str:
    # 1) secrets(openai 섹션) -> 2) secrets 루트 -> 3) env -> 4) default
    if "openai" in st.secrets and "OPENAI_MODEL" in st.secrets["openai"]:
        return st.secrets["openai"]["OPENAI_MODEL"]
    if "OPENAI_MODEL" in st.secrets:
        return st.secrets["OPENAI_MODEL"]
    return os.getenv("OPENAI_MODEL", "gpt-4o")

OPENAI_API_KEY = get_openai_api_key()
MODEL = get_openai_model()

# =========================
# Helpers
# =========================
TIME_PATTERNS = [
    r"^\s*\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.*)$",      # [00:12] 내용 / [00:01:12] 내용
    r"^\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–—]\s*(.*)$",  # 00:12 - 내용
]

def split_script_to_utterances(raw: str) -> List[Dict[str, Any]]:
    """
    사용자가 넣은 원문을 라인 단위 발언으로 분해.
    시간표기가 있으면 time에 저장.
    """
    lines = [ln.strip() for ln in raw.splitlines()]
    lines = [ln for ln in lines if ln]  # 빈줄 제거

    utterances: List[Dict[str, Any]] = []
    for i, ln in enumerate(lines, start=1):
        time_val = None
        text_val = ln

        for pat in TIME_PATTERNS:
            m = re.match(pat, ln)
            if m:
                time_val = m.group(1)
                text_val = m.group(2).strip()
                break

        utterances.append({
            "id": f"u{i}",
            "line_no": i,
            "time": time_val,
            "text": text_val
        })
    return utterances

def build_prompt(utterances: List[Dict[str, Any]]) -> str:
    """
    모델이 '제21조 위반 가능성'을 발언 단위로 판정하고,
    근거(조항/사유)를 구조화 JSON으로 내도록 유도.
    """
    items = []
    for u in utterances:
        tag = f"{u['id']} (line {u['line_no']}" + (f", time {u['time']}" if u["time"] else "") + ")"
        items.append(f"- {tag}: {u['text']}")
    joined = "\n".join(items)

    return f"""
너는 '금융소비자보호법 제21조(부당권유행위 금지)' 준수 점검을 돕는 내부 준법감시 보조 모델이다.
아래 상담/권유 발언 스크립트를 발언 단위로 분석하여, 제21조 위반 소지가 있는지 '가능성'을 판정하라.
주의: 법률 자문이 아니라 사전 스크리닝이며, 모호하면 '추가정보필요' 또는 '주의'로 처리한다.

중요: results 배열에는 각 utterance_id가 정확히 1번만 등장해야 한다.

[출력 형식: 반드시 JSON만 출력]
{{
  "summary": {{
    "has_violation": true/false,
    "risk_level": "HIGH" | "MEDIUM" | "LOW",
    "overall_note": "전체 요약/유의사항 2~4문장"
  }},
  "results": [
    {{
      "utterance_id": "u1",
      "verdict": "VIOLATION" | "CAUTION" | "CLEAR",
      "law_reference": "금융소비자보호법 제21조(부당권유행위 금지) - 해당되는 유형을 짧게",
      "reason": "구체적 사유(무엇이 왜 문제인지, 소비자 오인/압박/기만 요소 등)",
      "suggested_fix": "대체 표현/개선 권고(짧게)",
      "confidence": 0.0~1.0
    }}
  ]
}}

[판정 가이드]
- VIOLATION: 제21조 취지상 부당권유(허위/과장, 중요사항 누락, 손실가능성 은폐, 압박/강요, 오인 유발 등) 소지가 뚜렷
- CAUTION: 정보가 부족하거나 뉘앙스가 애매하여 추가 맥락 필요
- CLEAR: 위반 소지가 낮음

[분석 대상 발언]
{joined}
""".strip()

def call_openai_for_analysis(model: str, utterances: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    OpenAI Responses API 호출.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. Streamlit secrets 또는 환경변수를 확인해주세요.")

    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = build_prompt(utterances)

    resp = client.responses.create(
        model=model,
        input=prompt,
        text={"format": {"type": "json_object"}},
        temperature=0.2
    )

    raw = resp.output_text
    try:
        return json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"모델 응답 JSON 파싱 실패: {e}\n---raw---\n{raw}")

def verdict_to_style(verdict: str) -> str:
    if verdict == "VIOLATION":
        return "background-color:#ffdddd; color:#a40000; padding:2px 4px; border-radius:4px;"
    if verdict == "CAUTION":
        return "background-color:#fff3cd; color:#7a5a00; padding:2px 4px; border-radius:4px;"
    return ""  # CLEAR

def build_left_highlight_html(
    utterances: List[Dict[str, Any]],
    results_map: Dict[str, Dict[str, Any]],
    focus_id: Optional[str]
) -> str:
    """
    좌측 스크립트: 위반/주의 하이라이트 + focus_id면 굵은 테두리로 표시
    """
    rows = []
    for u in utterances:
        r = results_map.get(u["id"], {"verdict": "CLEAR"})
        verdict = r.get("verdict", "CLEAR")
        style = verdict_to_style(verdict)

        border = ""
        if focus_id and u["id"] == focus_id:
            border = "border:2px solid #333; padding:6px; border-radius:8px;"

        time_txt = f"[{u['time']}] " if u["time"] else ""
        meta = f"<span style='color:#888;'>({u['id']} / line {u['line_no']})</span>"

        line_html = f"""
        <div id="{u['id']}" style="margin:6px 0; {border}">
          <div style="font-size:14px;">
            <span style="{style}">{time_txt}{u['text']}</span>
            <span style="margin-left:8px;">{meta}</span>
          </div>
        </div>
        """
        rows.append(line_html)

    wrapper = "<div style='line-height:1.6;'>" + "\n".join(rows) + "</div>"
    return wrapper

def dedupe_results_by_utterance_id(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    모델이 동일 utterance_id를 중복 출력할 때 UI 충돌을 막기 위해 dedupe.
    뒤에 나온 결과를 우선으로 함.
    """
    dedup: Dict[str, Dict[str, Any]] = {}
    for r in results:
        uid = r.get("utterance_id")
        if uid:
            dedup[uid] = r
    return list(dedup.values())

# =========================
# Session State
# =========================
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "utterances" not in st.session_state:
    st.session_state.utterances = []
if "focus_id" not in st.session_state:
    st.session_state.focus_id = None
if "raw_script" not in st.session_state:
    st.session_state.raw_script = ""

# =========================
# UI
# =========================
st.title("금융소비자보호법 제21조 위반 점검(프로토타입)")

st.subheader("1) 스크립트 입력")

raw = st.text_area(
    "상담/권유 대화 스크립트를 붙여넣으세요 (라인 단위로 발언 분리). 시간표기가 있으면 [00:12] 형태 권장.",
    height=220,
    value=st.session_state.raw_script,
    placeholder="[00:01] 안녕하세요 고객님...\n[00:05] 이 상품은 원금 손실 가능성이 전혀 없습니다...\n..."
)

col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    analyze = st.button("분석하기", type="primary", use_container_width=True)

if analyze:
    st.session_state.raw_script = raw
    utterances = split_script_to_utterances(raw)
    st.session_state.utterances = utterances

    if not utterances:
        st.warning("분석할 발언이 없습니다. 텍스트를 입력해주세요.")
    else:
        with st.spinner("분석 중..."):
            try:
                analysis = call_openai_for_analysis(MODEL, utterances)
                st.session_state.analysis = analysis
                st.session_state.focus_id = None
            except Exception as e:
                st.error(str(e))

st.divider()
st.subheader("2) 분석 결과")

if not st.session_state.analysis:
    st.info("위 입력창에 스크립트를 넣고 **분석하기**를 눌러주세요.")
else:
    analysis = st.session_state.analysis
    utterances = st.session_state.utterances
    focus_id = st.session_state.focus_id

    results = analysis.get("results", [])
    results = dedupe_results_by_utterance_id(results)

    results_map = {r.get("utterance_id"): r for r in results if r.get("utterance_id")}

    left, right = st.columns([1.25, 1])

    # ---- LEFT: highlighted script
    with left:
        st.markdown("#### 스크립트(위반/주의 하이라이트)")
        html = build_left_highlight_html(utterances, results_map, focus_id)
        st.markdown(html, unsafe_allow_html=True)

        if focus_id:
            fu = next((u for u in utterances if u["id"] == focus_id), None)
            if fu:
                st.caption(
                    f"선택됨 → {fu['id']} / line {fu['line_no']}"
                    + (f" / time {fu['time']}" if fu["time"] else "")
                )

    # ---- RIGHT: reasons + jump
    with right:
        st.markdown("#### 위반 근거 / 발언별 조치")

        summ = analysis.get("summary", {})
        st.write(f"- **Risk Level**: {summ.get('risk_level', 'N/A')}")
        st.write(f"- **Note**: {summ.get('overall_note', '')}")

        st.markdown("---")

        for idx, r in enumerate(results):
            uid = r.get("utterance_id", "unknown")
            verdict = r.get("verdict", "CLEAR")
            law_ref = r.get("law_reference", "")
            reason = r.get("reason", "")
            fix = r.get("suggested_fix", "")
            conf = r.get("confidence", None)

            u = next((x for x in utterances if x["id"] == uid), None)
            line_meta = ""
            if u:
                line_meta = f"line {u['line_no']}" + (f", time {u['time']}" if u["time"] else "")

            badge = "✅ CLEAR" if verdict == "CLEAR" else ("⚠️ CAUTION" if verdict == "CAUTION" else "🛑 VIOLATION")

            with st.container(border=True):
                st.markdown(f"**{badge} — {uid} ({line_meta})**")
                if u:
                    st.write(u["text"])

                if law_ref:
                    st.markdown(f"- **법/조항**: {law_ref}")
                if reason:
                    st.markdown(f"- **구체 사유**: {reason}")
                if fix:
                    st.markdown(f"- **개선 권고**: {fix}")
                if conf is not None:
                    st.caption(f"confidence: {conf}")

                # key 유니크 보장(중복키 에러 방지)
                if st.button("이동하기", key=f"jump_{uid}_{idx}"):
                    st.session_state.focus_id = uid
                    st.rerun()

        st.markdown("---")

        st.markdown("#### 최종 판정")
        if any(r.get("verdict") == "VIOLATION" for r in results):
            st.error("고위험 준법 감시팀 알람 발생")
        else:
            st.success("CLEAR")

        st.caption("※ 본 결과는 자동 스크리닝이며, 최종 판단은 준법/법무 검토가 필요합니다.")
