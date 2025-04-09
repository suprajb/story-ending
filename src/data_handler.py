from pathlib import Path
from typing import List, Dict, Any

class DataHandler:
    """Handles loading and processing of story data"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True)
    
    def load_stories(self) -> List[Dict[str, Any]]:
        """
        Load stories from book chapters.
        
        Folder structure:
        - data/
          - <bookid>/
            - <bookid>-chapter/
              - chapter1.txt
              - chapter2.txt
              - ...
              - chapterN.txt
        
        Uses all chapters except the last one as the beginning.
        Uses the bookid as the story ID.
        
        Returns:
            List of story dictionaries, each containing:
                - id: book ID
                - beginning: combined text of all chapters except the last
        """
        stories = []
        
        for book_dir in self.data_dir.iterdir():
            if not book_dir.is_dir():
                continue
                
            book_id = book_dir.name
            chapters_dir = book_dir / f"{book_id}-chapters"
            
            if not chapters_dir.exists() or not chapters_dir.is_dir():
                print(f"Warning: No chapters directory found for book {book_id}")
                continue
                
            # sort chapter files
            chapter_files = sorted([f for f in chapters_dir.iterdir() if f.is_file()])
            
            if len(chapter_files) <= 1:
                print(f"Warning: Book {book_id} has too few chapters ({len(chapter_files)})")
                continue
                
            # last chapter is ignored for beginning
            story_chapters = chapter_files[:-1]
            
            #  combine beginning chapters
            beginning_text = ""
            for chapter_file in story_chapters:
                try:
                    with open(chapter_file, 'r', encoding='utf-8') as f:
                        chapter_text = f.read().strip()
                        beginning_text += chapter_text + "\n\n"
                except Exception as e:
                    print(f"Error reading chapter {chapter_file}: {e}")
            
            if beginning_text:
                stories.append({
                    "id": book_id,
                    "beginning": beginning_text.strip()
                })
                print(f"Loaded book {book_id} with {len(story_chapters)} chapters")
        
        print(f"Loaded {len(stories)} books from {self.data_dir}")
        return stories
