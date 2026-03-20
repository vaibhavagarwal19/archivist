from app.core.config import get_settings
from app.services.llm_client import get_llm_client


class SummaryService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = get_llm_client()

    async def summarize_meeting(self, meeting_notes: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarize meeting notes into: overview, decisions, action items, risks."
                    ),
                },
                {"role": "user", "content": meeting_notes},
            ],
        )
        return response.choices[0].message.content or ""
