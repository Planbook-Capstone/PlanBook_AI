from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query

router = APIRouter()


@router.post("/lesson-plan")
async def generate_lesson_plan():
    return {"message": "Hello"}
