def get_data(source: str, metric: str) -> str:
    return f"Fetched mock data from {source} for metric={metric}."


def send_email(recipient: str, subject: str, body: str) -> str:
    return f"Email queued to {recipient} with subject '{subject}'."


def create_ticket(team: str, issue: str, priority: str = "medium") -> str:
    return f"Ticket created for team={team} priority={priority}: {issue}"
