import asyncio
import json
import os
from typing import List
from groq import AsyncGroq
from src.models import MeetingExtraction

class AsyncTranscriptIngestor:
    def __init__(self, api_key: str = None, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = AsyncGroq(api_key=self.api_key) if self.api_key else None
        self.model = model

    async def extract_from_transcript(self, meeting_id: str, transcript_text: str) -> MeetingExtraction:
        if not self.client:
            raise ValueError("GROQ_API_KEY is not set.")

        prompt = f"""
        You are an expert GraphRAG data extraction agent. Analyze the following meeting transcript.
        Extract all actionable tasks, entities (people, teams, projects), and cross-functional relationships.
        You MUST respond strictly with a valid JSON object matching this exact structure:
        {{
          "meeting_id": "{meeting_id}",
          "action_items": [
            {{"task_id": "TASK_1", "description": "...", "assignee": "...", "deadline": "...", "status": "Pending", "dependencies": []}}
          ],
          "entities": [
            {{"name": "...", "label": "PERSON|TEAM|PROJECT"}}
          ],
          "relationships": [
            {{"source": "...", "target": "...", "relation": "ASSIGNED_TO|DEPENDS_ON|PART_OF"}}
          ]
        }}
        
        Transcript:
        {transcript_text}
        """
        
        try:
            response = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a precise data extraction engine. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            return MeetingExtraction(**data)
        except Exception as e:
            print(f"[Error] Failed to extract from {meeting_id}: {e}")
            return MeetingExtraction(meeting_id=meeting_id, action_items=[], entities=[], relationships=[])

    async def process_batch(self, transcripts: List[dict], batch_size: int = 10) -> List[MeetingExtraction]:
        results = []
        for i in range(0, len(transcripts), batch_size):
            batch = transcripts[i:i + batch_size]
            tasks = [self.extract_from_transcript(item["id"], item["text"]) for item in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            print(f"Processed batch {i // batch_size + 1} / {(len(transcripts) + batch_size - 1) // batch_size}")
        return results
