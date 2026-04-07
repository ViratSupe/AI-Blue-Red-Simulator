import os
import json
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()


class BlueTeamAgent:
    def __init__(self, agent_id="BT-AI"):
        self.agent_id = agent_id
        
        # ⚠️ TEMP: Replace with env variable later
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def analyse(self, log):
        prompt = f"""
        You are a cybersecurity SOC analyst.

        Analyze this log:
        {log}

        Determine:
        - threat_level (LOW, MEDIUM, HIGH)
        - reason
        - recommended_action (block_ip, monitor, ignore)

        Respond ONLY in JSON format:
        {{
            "threat_level": "...",
            "reason": "...",
            "recommended_action": "...",
            "source_ip": "{log.get("ip", "unknown")}",
            "source_log_id": "{log.get("log_id", "unknown")}"
        }}
        """

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json"
                )
            )

            text = response.text

            # 🔥 Safe JSON parsing
            try:
                result = json.loads(text)
            except:
                import re
                json_text = re.search(r"\{.*\}", text, re.DOTALL).group()
                result = json.loads(json_text)

            # ✅ ADD REQUIRED FIELDS (FIXES YOUR ERROR)
            result["assessed_by"] = self.agent_id
            result["assessed_at"] = datetime.utcnow().isoformat()

            return result

        except Exception as e:
            # 🔥 FULL SAFE FALLBACK (NO CRASH)
            return {
                "threat_level": "LOW",
                "reason": f"Error in AI detection: {str(e)}",
                "recommended_action": "monitor",
                "source_ip": log.get("ip", "unknown"),
                "source_log_id": log.get("log_id", "unknown"),
                "assessed_by": self.agent_id,
                "assessed_at": datetime.utcnow().isoformat()
            }