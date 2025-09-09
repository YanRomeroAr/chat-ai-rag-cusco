"""
Procesador de documentos TUPA para convertir PDFs a vectores
Este módulo se encarga de procesar y fragmentar documentos TUPA
"""

import os
import PyPDF2
import logging
from typing import List, Dict, Generator
import re
from dataclasses import dataclass

from rag_config import rag_config
from pinecone_client import pinecone_client

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Fragmento de documento procesado"""
    id: str
    text: str
    metadata: Dict

class DocumentProcessor:
    """Procesador de documentos TUPA"""
    
    def __init__(self):
        """Inicializa el procesador de documentos"""
        self.chunk_size = rag_config.chunk_size
        self.chunk_overlap = rag_config.chunk_overlap
        
        logger.info("✅ Procesador de documentos inicializado")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extrae texto de un archivo PDF
        
        Args:
            pdf_path: Ruta al archivo PDF
            
        Returns:
            Texto extraído del PDF
        """
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += f"\n--- Página {page_num + 1} ---\n{page_text}\n"
            
            logger.info(f"✅ Texto extraído de {pdf_path}: {len(text)} caracteres")
            return text
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo texto de {pdf_path}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """
        Limpia y normaliza el texto extraído
        
        Args:
            text: Texto crudo extraído
            
        Returns:
            Texto limpio y normalizado
        """
        # Remover caracteres extraños y normalizar espacios
        text = re.sub(r'\s+', ' ', text)  # Múltiples espacios a uno
        text = re.sub(r'\n+', '\n', text)  # Múltiples saltos de línea a uno
        text = text.replace('\x00', '')  # Remover caracteres null
        
        # Remover líneas muy cortas (probablemente headers/footers)
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 10]
        
        return '\n'.join(cleaned_lines)
    
    def chunk_text(self, text: str, source: str) -> List[DocumentChunk]:
        """
        Divide el texto en chunks manejables
        
        Args:
            text: Texto a dividir
            source: Fuente del documento
            
        Returns:
            Lista de DocumentChunk
        """
        chunks = []
        
        # Dividir por párrafos primero
        paragraphs = text.split('\n\n')
        current_chunk = ""
        chunk_id = 0
        
        for paragraph in paragraphs:
            # Si agregar este párrafo excede el tamaño máximo
            if len(current_chunk) + len(paragraph) > self.chunk_size:
                if current_chunk:  # Si hay contenido en el chunk actual
                    chunks.append(DocumentChunk(
                        id=f"{source}_chunk_{chunk_id}",
                        text=current_chunk.strip(),
                        metadata={
                            'source': source,
                            'chunk_id': chunk_id,
                            'document_type': 'tupa'
                        }
                    ))
                    chunk_id += 1
                    
                    # Iniciar nuevo chunk con overlap
                    if self.chunk_overlap > 0:
                        # Tomar las últimas palabras del chunk anterior
                        words = current_chunk.split()
                        overlap_words = words[-self.chunk_overlap:] if len(words) > self.chunk_overlap else words
                        current_chunk = ' '.join(overlap_words) + ' ' + paragraph
                    else:
                        current_chunk = paragraph
                else:
                    current_chunk = paragraph
            else:
                current_chunk += '\n\n' + paragraph if current_chunk else paragraph
        
        # Agregar el último chunk si tiene contenido
        if current_chunk.strip():
            chunks.append(DocumentChunk(
                id=f"{source}_chunk_{chunk_id}",
                text=current_chunk.strip(),
                metadata={
                    'source': source,
                    'chunk_id': chunk_id,
                    'document_type': 'tupa'
                }
            ))
        
        logger.info(f"📄 {source} dividido en {len(chunks)} chunks")
        return chunks
    
    def process_pdf_file(self, pdf_path: str) -> List[DocumentChunk]:
        """
        Procesa un archivo PDF completo
        
        Args:
            pdf_path: Ruta al archivo PDF
            
        Returns:
            Lista de chunks procesados
        """
        try:
            # Extraer nombre del archivo sin extensión
            source_name = os.path.basename(pdf_path).replace('.pdf', '')
            
            # Extraer texto
            raw_text = self.extract_text_from_pdf(pdf_path)
            if not raw_text:
                logger.warning(f"⚠️ No se pudo extraer texto de {pdf_path}")
                return []
            
            # Limpiar texto
            clean_text = self.clean_text(raw_text)
            
            # Dividir en chunks
            chunks = self.chunk_text(clean_text, source_name)
            
            logger.info(f"
