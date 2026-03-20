from sqlalchemy.orm import Session

from app.schemas.report import WorkflowResponse
from app.models.user import User
from app.services.internal_action_service import InternalActionService


class WorkflowService:
    def __init__(self) -> None:
        self.actions = InternalActionService()

    async def execute(
        self,
        db: Session,
        owner: User,
        objective: str,
        recipient_email: str,
    ) -> WorkflowResponse:
        tool_calls = [
            self.actions.get_data(db, owner, objective),
            self.actions.create_ticket(db, owner, "operations", objective, "high"),
            self.actions.send_email(
                db,
                owner,
                recipient_email,
                "Workflow execution completed",
                f"Objective completed: {objective}",
            ),
        ]
        return WorkflowResponse(status="completed", objective=objective, steps=tool_calls)
