import json

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.report import Report
from app.models.user import User
from app.models.workflow_action import WorkflowAction
from app.schemas.chat import ToolCallRecord


class InternalActionService:
    def get_data(self, db: Session, owner: User, metric: str) -> ToolCallRecord:
        document_count = db.scalar(
            select(func.count()).select_from(Document).where(Document.owner_id == owner.id)
        ) or 0
        report_count = db.scalar(
            select(func.count()).select_from(Report).where(Report.owner_id == owner.id)
        ) or 0
        recent_documents = db.scalars(
            select(Document.file_name)
            .where(Document.owner_id == owner.id)
            .order_by(Document.created_at.desc())
            .limit(3)
        ).all()

        payload = {
            "metric": metric,
            "document_count": int(document_count),
            "report_count": int(report_count),
            "recent_documents": recent_documents,
        }
        output = (
            f"Internal data snapshot for '{metric}': "
            f"{document_count} document(s), {report_count} report(s), "
            f"recent documents={', '.join(recent_documents) if recent_documents else 'none'}."
        )
        self._record_action(
            db,
            owner,
            action_type="get_data",
            title=f"Data lookup for {metric[:80]}",
            input_payload={"metric": metric},
            output_payload=payload,
        )
        return ToolCallRecord(
            name="get_data",
            args={"source": "internal-app-db", "metric": metric},
            output=output,
        )

    def create_ticket(self, db: Session, owner: User, team: str, issue: str, priority: str) -> ToolCallRecord:
        payload = {"team": team, "issue": issue, "priority": priority}
        action = self._record_action(
            db,
            owner,
            action_type="create_ticket",
            title=f"{team.title()} ticket",
            input_payload=payload,
            output_payload={**payload, "ticket_status": "open"},
        )
        return ToolCallRecord(
            name="create_ticket",
            args=payload,
            output=f"Ticket #{action.id} created for team={team} priority={priority}: {issue}",
        )

    def send_email(
        self,
        db: Session,
        owner: User,
        recipient: str,
        subject: str,
        body: str,
    ) -> ToolCallRecord:
        payload = {"recipient": recipient, "subject": subject, "body": body}
        action = self._record_action(
            db,
            owner,
            action_type="send_email",
            title=subject[:255] or "Workflow email",
            input_payload=payload,
            output_payload={**payload, "delivery_status": "queued"},
        )
        return ToolCallRecord(
            name="send_email",
            args=payload,
            output=f"Email #{action.id} queued to {recipient} with subject '{subject}'.",
        )

    def _record_action(
        self,
        db: Session,
        owner: User,
        action_type: str,
        title: str,
        input_payload: dict,
        output_payload: dict,
    ) -> WorkflowAction:
        action = WorkflowAction(
            action_type=action_type,
            title=title,
            status="completed",
            input_payload=json.dumps(input_payload, ensure_ascii=True),
            output_payload=json.dumps(output_payload, ensure_ascii=True),
            owner_id=owner.id,
        )
        db.add(action)
        db.commit()
        db.refresh(action)
        return action
