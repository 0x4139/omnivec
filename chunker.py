from semantic_text_splitter import MarkdownSplitter
from semantic_text_splitter.semantic_text_splitter import TextSplitter


class Chunker:

    def markdown(self, text,capacity,trim,overlap):
        splitter = MarkdownSplitter(capacity=capacity, trim=trim, overlap=overlap)
        return splitter.chunks(text)

    def text(self, text,capacity,trim,overlap):
        splitter =TextSplitter (capacity=capacity, trim=trim, overlap=overlap)
        return splitter.chunks(text)
