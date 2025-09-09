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
            
            logger.info(f"✅ Procesado {pdf_path}: {len(chunks)} chunks creados")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error procesando {pdf_path}: {e}")
            return []
    
    def process_multiple_pdfs(self, pdf_directory: str) -> List[DocumentChunk]:
        """
        Procesa múltiples archivos PDF de un directorio
        
        Args:
            pdf_directory: Directorio con archivos PDF
            
        Returns:
            Lista combinada de todos los chunks
        """
        all_chunks = []
        
        try:
            pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
            logger.info(f"📂 Encontrados {len(pdf_files)} archivos PDF en {pdf_directory}")
            
            for pdf_file in pdf_files:
                pdf_path = os.path.join(pdf_directory, pdf_file)
                chunks = self.process_pdf_file(pdf_path)
                all_chunks.extend(chunks)
            
            logger.info(f"✅ Procesados {len(pdf_files)} PDFs: {len(all_chunks)} chunks totales")
            return all_chunks
            
        except Exception as e:
            logger.error(f"❌ Error procesando directorio {pdf_directory}: {e}")
            return []
    
    def upload_chunks_to_pinecone(self, chunks: List[DocumentChunk]) -> bool:
        """
        Sube chunks procesados a Pinecone
        
        Args:
            chunks: Lista de DocumentChunk a subir
            
        Returns:
            True si fue exitoso
        """
        try:
            if not pinecone_client:
                logger.error("❌ Cliente Pinecone no disponible")
                return False
            
            # Convertir chunks a formato esperado por Pinecone
            documents = []
            for chunk in chunks:
                documents.append({
                    'id': chunk.id,
                    'text': chunk.text,
                    'metadata': chunk.metadata
                })
            
            # Subir a Pinecone
            success = pinecone_client.upsert_documents(documents)
            
            if success:
                logger.info(f"✅ {len(chunks)} chunks subidos a Pinecone exitosamente")
            else:
                logger.error("❌ Error subiendo chunks a Pinecone")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error subiendo a Pinecone: {e}")
            return False

def create_sample_tupa_documents() -> List[Dict]:
    """
    Crea documentos de ejemplo del TUPA para testing
    
    Returns:
        Lista de documentos de ejemplo
    """
    sample_docs = [
        {
            'id': 'tupa_licencia_funcionamiento',
            'text': """
            LICENCIA DE FUNCIONAMIENTO - GOBIERNO REGIONAL CUSCO
            
            REQUISITOS:
            1. Solicitud dirigida al Gerente Regional de Desarrollo Económico
            2. Copia del RUC y DNI del representante legal
            3. Declaración Jurada de cumplimiento de condiciones de seguridad
            4. Croquis de ubicación del establecimiento
            5. Copia de la autorización sectorial correspondiente
            
            PLAZO: 15 días hábiles
            COSTO: S/ 120.00 (12% UIT)
            
            HORARIOS DE ATENCIÓN:
            Lunes a Viernes: 8:00 AM - 4:00 PM
            Ubicación: Av. Regional 123, Cusco
            """,
            'metadata': {
                'source': 'TUPA_2024_Licencias',
                'document_type': 'tupa',
                'section': 'licencia_funcionamiento'
            }
        },
        {
            'id': 'tupa_permiso_construccion',
            'text': """
            PERMISO DE CONSTRUCCIÓN - GOBIERNO REGIONAL CUSCO
            
            MODALIDAD A - Construcciones menores (hasta 120 m²):
            REQUISITOS:
            1. Solicitud con firma del propietario
            2. Copia literal de dominio vigente
            3. Planos de arquitectura y estructuras (2 juegos)
            4. Memoria descriptiva firmada por ingeniero colegiado
            5. Estudio de mecánica de suelos
            
            PLAZO: 30 días hábiles
            COSTO: S/ 350.00 + S/ 2.50 por m²
            
            MODALIDAD B - Construcciones mayores (más de 120 m²):
            Requiere evaluación técnica adicional
            PLAZO: 45 días hábiles
            """,
            'metadata': {
                'source': 'TUPA_2024_Construccion',
                'document_type': 'tupa',
                'section': 'permiso_construccion'
            }
        },
        {
            'id': 'tupa_certificado_zonificacion',
            'text': """
            CERTIFICADO DE ZONIFICACIÓN Y VÍAS - GOBIERNO REGIONAL CUSCO
            
            DESCRIPCIÓN:
            Documento que certifica la zonificación urbana y características de vías
            de un predio específico según el Plan de Desarrollo Urbano vigente.
            
            REQUISITOS:
            1. Solicitud dirigida a la Gerencia de Infraestructura
            2. Copia simple del documento de identidad del solicitante
            3. Copia literal de dominio no mayor a 30 días
            4. Plano de ubicación y localización del predio
            
            PLAZO: 10 días hábiles
            COSTO: S/ 85.00 (8.5% UIT)
            
            VALIDEZ: 2 años desde su emisión
            
            OFICINA DE ATENCIÓN:
            Gerencia Regional de Infraestructura
            Jr. Comercio 456, 2do piso, Cusco
            Teléfono: (084) 234-567
            """,
            'metadata': {
                'source': 'TUPA_2024_Zonificacion',
                'document_type': 'tupa',
                'section': 'certificado_zonificacion'
            }
        }
    ]
    
    return sample_docs

# Instancia global del procesador
document_processor = DocumentProcessor()
