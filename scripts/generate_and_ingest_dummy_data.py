
import sys
import os
import random
import uuid

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.endee_client import EndeeVectorDB
from backend.core.embeddings import EmbeddingService
from backend.config import get_settings

# Real-looking dummy data
REAL_dummy_papers = [
    {
        "title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
        "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        "year": 2017,
        "category": "cs.CL",
        "url": "https://arxiv.org/abs/1706.03762"
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models (Peters et al., 2018a; Radford et al., 2018), BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.",
        "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
        "year": 2018,
        "category": "cs.CL",
        "url": "https://arxiv.org/abs/1810.04805"
    },
    {
        "title": "Generative Adversarial Nets",
        "abstract": "We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G.",
        "authors": ["Ian J. Goodfellow", "Jean Pouget-Abadie", "Mehdi Mirza"],
        "year": 2014,
        "category": "stat.ML",
        "url": "https://arxiv.org/abs/1406.2661"
    },
    {
        "title": "ImageNet Classification with Deep Convolutional Neural Networks",
        "abstract": "We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 37.5% and 17.0% which is considerably better than the previous state-of-the-art.",
        "authors": ["Alex Krizhevsky", "Ilya Sutskever", "Geoffrey E. Hinton"],
        "year": 2012,
        "category": "cs.CV",
        "url": "https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks"
    },
    {
        "title": "Deep Residual Learning for Image Recognition",
        "abstract": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions.",
        "authors": ["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren", "Jian Sun"],
        "year": 2015,
        "category": "cs.CV",
        "url": "https://arxiv.org/abs/1512.03385"
    },
    {
        "title": "Visualizing and Understanding Convolutional Networks",
        "abstract": "Large Convolutional Network models have recently demonstrated impressive classification performance on the ImageNet benchmark Krizhevsky et al. [18]. However there is no clear understanding of why they perform so well, or how they might be improved. In this paper we address both issues.",
        "authors": ["Matthew D. Zeiler", "Rob Fergus"],
        "year": 2013,
        "category": "cs.CV",
        "url": "https://arxiv.org/abs/1311.2901"
    },
    {
        "title": "Adam: A Method for Stochastic Optimization",
        "abstract": "We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or parameters.",
        "authors": ["Diederik P. Kingma", "Jimmy Ba"],
        "year": 2014,
        "category": "cs.LG",
        "url": "https://arxiv.org/abs/1412.6980"
    },
    {
        "title": "Sequence to Sequence Learning with Neural Networks",
        "abstract": "Deep Neural Networks (DNNs) are powerful models that have achieved excellent performance on difficult learning tasks. Although DNNs work well whenever large labeled training sets are available, they cannot be used to map sequences to sequences. In this paper, we present a general end-to-end approach to sequence learning that makes minimal assumptions on the sequence structure.",
        "authors": ["Ilya Sutskever", "Oriol Vinyals", "Quoc V. Le"],
        "year": 2014,
        "category": "cs.CL",
        "url": "https://arxiv.org/abs/1409.3215"
    },
    {
        "title": "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
        "abstract": "Deep neural nets with a large number of parameters are very powerful machine learning systems. However, overfitting is a serious problem in such networks. Large networks are also slow to use, making it difficult to deal with overfitting by combining the predictions of many different large neural nets at test time. Dropout is a technique for addressing this problem.",
        "authors": ["Nitish Srivastava", "Geoffrey Hinton", "Alex Krizhevsky", "Ilya Sutskever"],
        "year": 2014,
        "category": "cs.LG",
        "url": "http://jmlr.org/papers/v15/srivastava14a.html"
    },
    {
        "title": "Playing Atari with Deep Reinforcement Learning",
        "abstract": "We present the first deep learning model to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning. The model is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards.",
        "authors": ["Volodymyr Mnih", "Koray Kavukcuoglu", "David Silver"],
        "year": 2013,
        "category": "cs.LG",
        "url": "https://arxiv.org/abs/1312.5602"
    },
    {
        "title": "Chatbots regarding Artificial Intelligence",
        "abstract": "This paper explores the architecture and implementation of chatbots powered by large language models, focusing on their ability to understand context and generate human-like responses in various domains.",
        "authors": ["Jane Smith", "John Doe"],
        "year": 2023,
        "category": "cs.AI",
        "url": "https://example.com/chatbot-ai"
    },
    {
        "title": "Healthcare applications of Machine Learning",
        "abstract": "A comprehensive review of how machine learning algorithms (SVM, Random Forest, Neural Networks) are applied in diagnostic imaging, patient outcome prediction, and personalized medicine.",
        "authors": ["Alice Johnson", "Bob Williams"],
        "year": 2024,
        "category": "cs.AI",
        "url": "https://example.com/health-ml"
    },
     {
        "title": "Large Language Models in Code Generation",
        "abstract": "Evaluating the performance of GPT-4 and CodeLlama on complex software engineering tasks, specifically focusing on unit test generation and refactoring legacy codebases.",
        "authors": ["David Chen", "Emily Wu"],
        "year": 2024,
        "category": "cs.SE",
        "url": "https://example.com/llm-code"
    },
    {
        "title": "Efficiency in Vector Databases",
        "abstract": "Comparing HNSW and IVF indices for high-dimensional vector retrieval. We analyze trade-offs between recall and latency for billion-scale datasets.",
        "authors": ["Michael Brown", "Sarah Davis"],
        "year": 2023,
        "category": "cs.DB",
        "url": "https://example.com/vector-db"
    },
    {
        "title": "Reinforcement Learning from Human Feedback",
        "abstract": "Aligning language models with human intent using RLHF. We discuss the reward modeling process and PPO optimization stability.",
        "authors": ["OpenAI Team"],
        "year": 2022,
        "category": "cs.LG",
        "url": "https://example.com/rlhf"
    }

]


def main():
    print("Starting dummy data ingestion...")
    settings = get_settings()
    
    # Initialize services
    endee = EndeeVectorDB(url=settings.ENDEE_URL)
    embeddings = EmbeddingService(model_name=settings.EMBEDDING_MODEL)
    
    print(f"Found {len(REAL_dummy_papers)} papers to ingest.")
    
    batch_papers = []
    
    # Process papers
    for p in REAL_dummy_papers:
        paper_id = str(uuid.uuid4())
        
        # Create text for embedding
        text_to_embed = f"{p['title']} {p['abstract']}"
        
        # Generate embedding (this uses our deterministic fallback if torch is broken)
        # Note: EmbeddingService.embed_text returns a single list
        vector = embeddings.embed_text(text_to_embed)
        
        paper_doc = {
            "id": paper_id,
            "vector": vector, # In a real system this would be the key for vector search
            "meta": {
                "title": p["title"],
                "abstract": p["abstract"],
                "authors": p["authors"],
                "url": p["url"]
            },
            "filter": {
                "year": p["year"],
                "category": p["category"],
                "citations": random.randint(10, 50000)
            }
        }
        batch_papers.append(paper_doc)
        
    # Upsert to 'In-Memory' DB
    print(f"Upserting {len(batch_papers)} vectors to Endee...")
    # We call the upsert method which we will modify to store data in-memory
    endee.upsert_vectors("research_papers_dense", batch_papers)
    
    print("Ingestion complete! You can now search for these papers.")

if __name__ == "__main__":
    main()
