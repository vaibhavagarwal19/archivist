from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.report import Report
from app.models.user import User
from app.services.llm_client import get_llm_client


class ReportService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = get_llm_client()

    async def generate_report(self, db: Session, owner: User, title: str, prompt: str) -> Report:
        response = await self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {"role": "system", "content": "Create a concise professional internal report."},
                {"role": "user", "content": prompt},
            ],
        )
        report = Report(
            title=title,
            prompt=prompt,
            content=response.choices[0].message.content or "",
            owner_id=owner.id,
        )
        db.add(report)
        db.commit()
        db.refresh(report)
        return report
