# tests/test_file_handler.py

import unittest
from file_handler import read_pdf, read_docx, read_txt, chunk_text

class TestFileHandler(unittest.TestCase):
    def test_read_txt(self):
        # Create a sample text file
        sample_text = "This is a sample text file."
        with open("test_sample.txt", "w", encoding='utf-8') as f:
            f.write(sample_text)
        
        result = read_txt("test_sample.txt")
        self.assertEqual(result, sample_text)
        
        # Clean up
        import os
        os.remove("test_sample.txt")

    # def test_chunk_text(self):
    #     text = "Sentence one. Sentence two. Sentence three."
    #     expected_chunks = [
    #         "Sentence one.",
    #         "Sentence two.",
    #         "Sentence three."
    #     ]
    #     result = chunk_text(text, max_length=1000, chunk_overlap=100)
    #     self.assertEqual(result, expected_chunks)
    
    #     # Test with smaller max_length and appropriate chunk_overlap
    #     expected_chunks_small = [
    #         "Sentence one.",
    #         "Sentence two.",
    #         "Sentence three."
    #     ]
    #     result_small = chunk_text(text, max_length=15, chunk_overlap=5)  # Set chunk_overlap < max_length
    #     self.assertEqual(result_small, expected_chunks_small)
    
    # ... [Other test functions] ...

if __name__ == '__main__':
    unittest.main()