"""
Cliente de Pinecone para gestión de vectores - Compatible con API 2.2.4
Este módulo maneja toda la interacción con Pinecone usando la API antigua
"""

import pinecone
import logging
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

from rag_config import rag_config

logger = logging.getLogger(__name__)

class PineconeClient:
    """Cliente para interactuar con Pinecone"""
    
    def __init__(self):
        """Inicializa el cliente de Pinecone"""
        try:
            # Inicializar Pinecone con API antigua
            pinecone.init(
                api_key=rag_config.pinecone_api_key,
                environment=rag_config.pinecone_environment
            )
            
            # Cargar modelo de embeddings
            self.embedding_model = SentenceTransformer(rag_config.embedding_model)
            logger.info(f"✅ Modelo de embeddings cargado: {rag_config.embedding_model}")
            
            # Conectar al índice
            self.index = None
            self._connect_to_index()
            
        except Exception as e:
            logger.error(f"❌ Error inicializando Pinecone: {e}")
            raise
    
    def _connect_to_index(self):
        """Conecta al índice de Pinecone, lo crea si no existe"""
        try:
            # Verificar si el índice existe
            if rag_config.pinecone_index_name not in pinecone.list_indexes():
                logger.info(f"Creando índice: {rag_config.pinecone_index_name}")
                
                # Crear índice nuevo con API antigua
                pinecone.create_index(
                    name=rag_config.pinecone_index_name,
                    dimension=rag_config.embedding_dimension,
                    metric='cosine',  # Métrica de similitud coseno
                    metadata_config={
                        "indexed": ["document_type", "section", "source"]
                    }
                )
                logger.info("✅ Índice creado exitosamente")
            
            # Conectar al índice
            self.index = pinecone.Index(rag_config.pinecone_index_name)
            logger.info(f"✅ Conectado al índice: {rag_config.pinecone_index_name}")
            
        except Exception as e:
            logger.error(f"❌ Error conectando al índice: {e}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Genera embedding para un texto
        
        Args:
            text: Texto a convertir en embedding
            
        Returns:
            Lista de floats representando el embedding
        """
        try:
            # Generar embedding usando sentence-transformers
            embedding = self.embedding_model.encode(text)
            
            # Convertir a lista de Python (Pinecone requiere esto)
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"❌ Error generando embedding: {e}")
            raise
    
    def upsert_documents(self, documents: List[Dict]) -> bool:
        """
        Inserta o actualiza documentos en Pinecone
        
        Args:
            documents: Lista de documentos con formato:
                [{
                    'id': 'doc_id',
                    'text': 'contenido del documento',
                    'metadata': {'source': 'filename', 'section': 'chapter1'}
                }]
                
        Returns:
            True si fue exitoso
        """
        try:
            vectors_to_upsert = []
            
            for doc in documents:
                # Generar embedding para el texto
                embedding = self.generate_embedding(doc['text'])
                
                # Preparar vector para Pinecone
                vector = {
                    'id': doc['id'],
                    'values': embedding,
                    'metadata': {
                        'text': doc['text'],  # Guardamos el texto original
                        **doc.get('metadata', {})
                    }
                }
                vectors_to_upsert.append(vector)
            
            # Insertar en lotes de 100 (límite de Pinecone)
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch)
                logger.info(f"📤 Insertados {len(batch)} vectores (lote {i//batch_size + 1})")
            
            logger.info(f"✅ {len(documents)} documentos insertados exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error insertando documentos: {e}")
            return False
    
    def search_similar_documents(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Busca documentos similares a una consulta
        
        Args:
            query: Consulta del usuario
            top_k: Número de documentos a retornar
            
        Returns:
            Lista de documentos relevantes con scores
        """
        try:
            top_k = top_k or rag_config.top_k_results
            
            # Generar embedding de la consulta
            query_embedding = self.generate_embedding(query)
            
            # Buscar en Pinecone
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                include_values=False  # No necesitamos los vectores de vuelta
            )
            
            # Formatear resultados
            documents = []
            for match in search_results['matches']:
                # Filtrar por umbral de similitud
                if match['score'] >= rag_config.similarity_threshold:
                    documents.append({
                        'id': match['id'],
                        'text': match['metadata']['text'],
                        'score': match['score'],
                        'metadata': {k: v for k, v in match['metadata'].items() if k != 'text'}
                    })
            
            logger.info(f"🔍 Encontrados {len(documents)} documentos relevantes para: '{query[:50]}...'")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error buscando documentos: {e}")
            return []
    
    def get_index_stats(self) -> Dict:
        """
        Obtiene estadísticas del índice
        
        Returns:
            Diccionario con estadísticas
        """
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats.get('total_vector_count', 0),
                'dimension': stats.get('dimension', 0),
                'index_fullness': stats.get('index_fullness', 0)
            }
        except Exception as e:
            logger.error(f"❌ Error obteniendo estadísticas: {e}")
            return {}
    
    def delete_all_vectors(self) -> bool:
        """
        Elimina todos los vectores del índice (usar con cuidado)
        
        Returns:
            True si fue exitoso
        """
        try:
            self.index.delete(delete_all=True)
            logger.warning("🗑️ Todos los vectores eliminados del índice")
            return True
        except Exception as e:
            logger.error(f"❌ Error eliminando vectores: {e}")
            return False

# Instancia global del cliente
try:
    pinecone_client = PineconeClient()
    logger.info("✅ Cliente Pinecone inicializado")
except Exception as e:
    logger.error(f"❌ Error inicializando cliente Pinecone: {e}")
    pinecone_client = None
