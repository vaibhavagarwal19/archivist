from app.tasks.celery_app import celery_app


@celery_app.task(name="documents.post_process")
def post_process_document(document_id: int) -> dict:
    return {"status": "processed", "document_id": document_id}
