"""
Automated Document Generation System
Generates large volumes of diverse documents for knowledge base expansion.
"""

import os
import json
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
import string


@dataclass
class DocumentTemplate:
    """Template for generating documents."""
    category: str
    topics: List[str]
    sentence_templates: List[str]
    fact_templates: List[str]
    question_templates: List[str]


class DocumentGenerator:
    """
    Generates diverse documents programmatically for large-scale knowledge base.
    """
    
    def __init__(self, output_dir: str = "data/knowledge_base/generated"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, DocumentTemplate]:
        """Initialize document templates across diverse categories."""
        templates = {}
        
        # Technology templates
        templates['technology'] = DocumentTemplate(
            category='technology',
            topics=['AI', 'machine learning', 'blockchain', 'cloud computing', 'cybersecurity', 'IoT', '5G', 'quantum computing'],
            sentence_templates=[
                "{topic} is revolutionizing the way we approach modern challenges.",
                "The development of {topic} has accelerated in recent years.",
                "Applications of {topic} span across multiple industries.",
                "Experts predict that {topic} will transform business operations.",
                "Research in {topic} continues to advance rapidly.",
                "The impact of {topic} on society is significant and growing.",
                "Implementing {topic} requires careful planning and expertise.",
                "The future of {topic} looks promising with ongoing innovations.",
                "Challenges in {topic} include security, scalability, and adoption.",
                "Benefits of {topic} include efficiency, automation, and cost reduction."
            ],
            fact_templates=[
                "{topic} was first developed in the {decade}.",
                "The global market for {topic} is expected to reach {number} billion by {year}.",
                "Major companies investing in {topic} include {company1}, {company2}, and {company3}.",
                "Key applications of {topic} include {application1}, {application2}, and {application3}.",
                "The main challenges facing {topic} adoption are {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main applications of {topic}?",
                "How does {topic} work in practice?",
                "What are the benefits of implementing {topic}?",
                "What challenges does {topic} face?",
                "What is the future outlook for {topic}?"
            ]
        )
        
        # Science templates
        templates['science'] = DocumentTemplate(
            category='science',
            topics=['genetics', 'neuroscience', 'quantum physics', 'climate science', 'space exploration', 'marine biology', 'nanotechnology'],
            sentence_templates=[
                "Recent discoveries in {topic} have expanded our understanding of the natural world.",
                "Scientists are making breakthroughs in {topic} research.",
                "The study of {topic} has important implications for future technologies.",
                "Research in {topic} requires advanced equipment and methodologies.",
                "Applications of {topic} research include medicine, engineering, and environmental science.",
                "The complexity of {topic} continues to challenge researchers.",
                "Collaboration in {topic} research spans across international borders.",
                "Funding for {topic} research has increased significantly.",
                "Public interest in {topic} has grown in recent years.",
                "The potential of {topic} to solve global problems is substantial."
            ],
            fact_templates=[
                "The first major discovery in {topic} occurred in {year}.",
                "Current research in {topic} focuses on {focus_area}.",
                "Leading institutions in {topic} include {institution1} and {institution2}.",
                "The main tools used in {topic} research are {tool1} and {tool2}.",
                "Key challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the latest discoveries in {topic}?",
                "How does {topic} impact our daily lives?",
                "What are the main research areas in {topic}?",
                "What tools are used in {topic} research?",
                "What are the future directions of {topic}?"
            ]
        )
        
        # Business templates
        templates['business'] = DocumentTemplate(
            category='business',
            topics=['marketing', 'finance', 'operations', 'human resources', 'strategy', 'entrepreneurship', 'supply chain', 'e-commerce'],
            sentence_templates=[
                "Effective {topic} strategies are essential for business success.",
                "Companies are increasingly focusing on {topic} to gain competitive advantage.",
                "The field of {topic} has evolved significantly with digital transformation.",
                "Best practices in {topic} include data-driven decision making.",
                "Investing in {topic} can lead to significant returns.",
                "Challenges in {topic} management require innovative solutions.",
                "The role of {topic} in organizational success cannot be overstated.",
                "Modern approaches to {topic} emphasize automation and efficiency.",
                "Global trends in {topic} reflect changing market conditions.",
                "Expertise in {topic} is highly valued in today's business environment."
            ],
            fact_templates=[
                "The global market for {topic} services is valued at {number} billion.",
                "Leading companies in {topic} include {company1} and {company2}.",
                "Key metrics in {topic} include {metric1} and {metric2}.",
                "The adoption of {topic} technologies has increased by {percentage}%.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the key strategies for {topic}?",
                "How can businesses improve their {topic}?",
                "What are the latest trends in {topic}?",
                "What metrics should be tracked in {topic}?",
                "How does {topic} impact business performance?"
            ]
        )
        
        # Health templates
        templates['health'] = DocumentTemplate(
            category='health',
            topics=['nutrition', 'mental health', 'fitness', 'preventive care', 'chronic disease management', 'healthcare technology', 'public health'],
            sentence_templates=[
                "Proper {topic} is essential for maintaining overall health and well-being.",
                "Research has shown the importance of {topic} in disease prevention.",
                "Healthcare professionals emphasize the role of {topic} in patient care.",
                "Advances in {topic} have improved health outcomes significantly.",
                "Public awareness of {topic} has increased in recent years.",
                "The impact of {topic} on quality of life is substantial.",
                "Healthcare systems are investing in {topic} programs.",
                "Personalized approaches to {topic} are becoming more common.",
                "The relationship between {topic} and other health factors is complex.",
                "Future developments in {topic} promise to transform healthcare delivery."
            ],
            fact_templates=[
                "Studies show that {topic} can reduce the risk of {disease} by {percentage}%.",
                "The global cost of {topic}-related healthcare is {number} billion.",
                "Leading organizations in {topic} include {org1} and {org2}.",
                "Key recommendations for {topic} include {recommendation1} and {recommendation2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the benefits of {topic}?",
                "How can individuals improve their {topic}?",
                "What are the latest research findings on {topic}?",
                "What are the common misconceptions about {topic}?",
                "How does {topic} affect overall health?"
            ]
        )
        
        # Education templates
        templates['education'] = DocumentTemplate(
            category='education',
            topics=['online learning', 'STEM education', 'early childhood education', 'higher education', 'educational technology', 'curriculum design', 'special education'],
            sentence_templates=[
                "Modern approaches to {topic} emphasize student-centered learning.",
                "The integration of technology has transformed {topic} significantly.",
                "Research supports the effectiveness of {topic} in improving outcomes.",
                "Educators are adopting innovative methods in {topic}.",
                "The importance of {topic} in lifelong learning is well-established.",
                "Challenges in {topic} include accessibility and equity.",
                "Global trends in {topic} reflect changing workforce needs.",
                "Assessment methods in {topic} have evolved over time.",
                "Collaboration is key to successful {topic} implementation.",
                "The future of {topic} will be shaped by technological advances."
            ],
            fact_templates=[
                "Enrollment in {topic} programs has increased by {percentage}%.",
                "The average cost of {topic} is {number} per student.",
                "Leading institutions in {topic} include {institution1} and {institution2}.",
                "Key skills developed through {topic} include {skill1} and {skill2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the benefits of {topic}?",
                "How can educators improve {topic}?",
                "What are the latest trends in {topic}?",
                "What skills are developed through {topic}?",
                "How does {topic} impact learning outcomes?"
            ]
        )
        
        # Environment templates
        templates['environment'] = DocumentTemplate(
            category='environment',
            topics=['renewable energy', 'climate change mitigation', 'sustainable agriculture', 'waste management', 'biodiversity conservation', 'water conservation', 'green building'],
            sentence_templates=[
                "Sustainable practices in {topic} are crucial for environmental protection.",
                "Global initiatives in {topic} have gained momentum in recent years.",
                "The impact of {topic} on ecosystems is significant.",
                "Innovations in {topic} are making environmental solutions more accessible.",
                "Public awareness of {topic} has increased substantially.",
                "Economic benefits of {topic} include job creation and cost savings.",
                "Policy frameworks for {topic} are evolving globally.",
                "Technology plays a key role in advancing {topic} solutions.",
                "Community engagement is essential for successful {topic} implementation.",
                "The future of {topic} depends on continued innovation and investment."
            ],
            fact_templates=[
                "Global investment in {topic} reached {number} billion last year.",
                "The adoption of {topic} has reduced emissions by {percentage}%.",
                "Leading countries in {topic} include {country1} and {country2}.",
                "Key benefits of {topic} include {benefit1} and {benefit2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main benefits of {topic}?",
                "How can communities implement {topic}?",
                "What are the latest innovations in {topic}?",
                "What policies support {topic}?",
                "How does {topic} impact the environment?"
            ]
        )
        
        # Arts templates
        templates['arts'] = DocumentTemplate(
            category='arts',
            topics=['painting', 'sculpture', 'music', 'literature', 'theater', 'dance', 'film', 'photography', 'architecture', 'digital art'],
            sentence_templates=[
                "The art of {topic} has evolved significantly throughout history.",
                "Contemporary {topic} reflects modern cultural values and concerns.",
                "Masters of {topic} have created timeless works that inspire generations.",
                "The techniques used in {topic} require years of practice and dedication.",
                "Cultural influences shape the development of {topic} across different regions.",
                "Technology has transformed how {topic} is created and experienced.",
                "Educational programs in {topic} emphasize both technique and creativity.",
                "The market for {topic} has grown significantly in recent years.",
                "Preservation of {topic} heritage is important for future generations.",
                "Collaboration in {topic} often leads to innovative and unique works."
            ],
            fact_templates=[
                "The history of {topic} dates back to {period}.",
                "Famous practitioners of {topic} include {artist1} and {artist2}.",
                "The global market for {topic} is valued at {number} billion.",
                "Key techniques in {topic} include {technique1} and {technique2}.",
                "Major movements in {topic} include {movement1} and {movement2}."
            ],
            question_templates=[
                "What are the main techniques used in {topic}?",
                "How has {topic} evolved over time?",
                "Who are the most influential figures in {topic}?",
                "What are the current trends in {topic}?",
                "How does technology impact {topic}?"
            ]
        )
        
        # Law templates
        templates['law'] = DocumentTemplate(
            category='law',
            topics=['constitutional law', 'criminal law', 'civil law', 'international law', 'corporate law', 'intellectual property', 'environmental law', 'human rights'],
            sentence_templates=[
                "The principles of {topic} are fundamental to a just society.",
                "Recent developments in {topic} reflect changing social values.",
                "Legal professionals specializing in {topic} are in high demand.",
                "The interpretation of {topic} varies across different jurisdictions.",
                "Technology presents new challenges for {topic} regulation.",
                "International cooperation is essential for effective {topic} enforcement.",
                "Education in {topic} emphasizes both theory and practical application.",
                "The complexity of {topic} requires specialized expertise.",
                "Public awareness of {topic} rights has increased significantly.",
                "Future trends in {topic} will be shaped by technological and social changes."
            ],
            fact_templates=[
                "The first major legislation in {topic} was passed in {year}.",
                "Leading law schools for {topic} include {school1} and {school2}.",
                "The number of {topic} cases has increased by {percentage}%.",
                "Key principles of {topic} include {principle1} and {principle2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the key principles of {topic}?",
                "How does {topic} affect everyday life?",
                "What are the current challenges in {topic}?",
                "How is {topic} enforced?",
                "What are the future trends in {topic}?"
            ]
        )
        
        # Engineering templates
        templates['engineering'] = DocumentTemplate(
            category='engineering',
            topics=['civil engineering', 'mechanical engineering', 'electrical engineering', 'chemical engineering', 'software engineering', 'biomedical engineering', 'aerospace engineering'],
            sentence_templates=[
                "The field of {topic} is essential for modern infrastructure and technology.",
                "Innovations in {topic} have transformed how we live and work.",
                "Engineering projects in {topic} require careful planning and execution.",
                "Safety standards in {topic} are critical for public protection.",
                "Sustainability is becoming increasingly important in {topic} design.",
                "Collaboration is key to successful {topic} projects.",
                "Education in {topic} combines theoretical knowledge with practical skills.",
                "The demand for {topic} professionals continues to grow.",
                "Research in {topic} drives technological advancement.",
                "Global challenges require innovative {topic} solutions."
            ],
            fact_templates=[
                "The first major breakthrough in {topic} occurred in {year}.",
                "The global market for {topic} services is valued at {number} billion.",
                "Leading companies in {topic} include {company1} and {company2}.",
                "Key tools used in {topic} include {tool1} and {tool2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main applications of {topic}?",
                "How has {topic} evolved over time?",
                "What are the current challenges in {topic}?",
                "What skills are needed for {topic}?",
                "What are the future trends in {topic}?"
            ]
        )
        
        # Finance templates
        templates['finance'] = DocumentTemplate(
            category='finance',
            topics=['investment banking', 'asset management', 'corporate finance', 'risk management', 'financial planning', 'fintech', 'cryptocurrency', 'international finance'],
            sentence_templates=[
                "The field of {topic} plays a crucial role in the global economy.",
                "Technological innovations are transforming {topic} practices.",
                "Risk management is essential in {topic} operations.",
                "Regulatory frameworks for {topic} continue to evolve.",
                "Global markets are increasingly interconnected through {topic}.",
                "Data analytics is becoming central to {topic} decision-making.",
                "Sustainable investing is gaining prominence in {topic}.",
                "Education in {topic} requires both quantitative and analytical skills.",
                "The complexity of {topic} demands specialized expertise.",
                "Future trends in {topic} will be shaped by technology and regulation."
            ],
            fact_templates=[
                "The global market for {topic} is valued at {number} trillion.",
                "Leading firms in {topic} include {firm1} and {firm2}.",
                "The adoption of {topic} technologies has increased by {percentage}%.",
                "Key metrics in {topic} include {metric1} and {metric2}.",
                "Major risks in {topic} include {risk1} and {risk2}."
            ],
            question_templates=[
                "What are the main functions of {topic}?",
                "How does technology impact {topic}?",
                "What are the current trends in {topic}?",
                "What are the main risks in {topic}?",
                "How is {topic} regulated?"
            ]
        )
        
        # Media templates
        templates['media'] = DocumentTemplate(
            category='media',
            topics=['journalism', 'social media', 'broadcasting', 'digital media', 'content creation', 'media ethics', 'media law', 'advertising'],
            sentence_templates=[
                "The landscape of {topic} has been transformed by digital technology.",
                "Ethical considerations are increasingly important in {topic}.",
                "Audience engagement is crucial for successful {topic}.",
                "The business model of {topic} continues to evolve.",
                "Social media has significantly impacted {topic} practices.",
                "Quality journalism in {topic} requires dedication and resources.",
                "The reach of {topic} extends globally through digital platforms.",
                "Regulation of {topic} varies across different jurisdictions.",
                "Media literacy is essential for navigating modern {topic}.",
                "The future of {topic} will be shaped by technological innovation."
            ],
            fact_templates=[
                "The global audience for {topic} is estimated at {number} billion.",
                "Leading companies in {topic} include {company1} and {company2}.",
                "Revenue in {topic} has grown by {percentage}% annually.",
                "Key platforms in {topic} include {platform1} and {platform2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "How has technology transformed {topic}?",
                "What are the ethical considerations in {topic}?",
                "What are the current trends in {topic}?",
                "How is {topic} regulated?",
                "What skills are needed for {topic}?"
            ]
        )
        
        # Sports templates
        templates['sports'] = DocumentTemplate(
            category='sports',
            topics=['football', 'basketball', 'tennis', 'swimming', 'athletics', 'gymnastics', 'martial arts', 'esports'],
            sentence_templates=[
                "The sport of {topic} requires dedication, skill, and physical fitness.",
                "Professional {topic} has grown in popularity and commercial value.",
                "Training for {topic} involves both physical and mental preparation.",
                "Technology has enhanced training and performance in {topic}.",
                "International competitions in {topic} showcase global talent.",
                "Youth participation in {topic} promotes healthy lifestyles.",
                "Coaching in {topic} requires specialized knowledge and experience.",
                "The rules of {topic} are designed to ensure fair competition.",
                "Injuries in {topic} require proper prevention and treatment.",
                "The future of {topic} will be shaped by innovation and global participation."
            ],
            fact_templates=[
                "The global audience for {topic} is estimated at {number} billion.",
                "Top athletes in {topic} can earn up to {number} million annually.",
                "The first major {topic} competition was held in {year}.",
                "Key skills in {topic} include {skill1} and {skill2}.",
                "Major competitions in {topic} include {competition1} and {competition2}."
            ],
            question_templates=[
                "What are the key skills needed for {topic}?",
                "How has technology impacted {topic}?",
                "What are the major competitions in {topic}?",
                "How can athletes prevent injuries in {topic}?",
                "What are the current trends in {topic}?"
            ]
        )
        
        # Travel templates
        templates['travel'] = DocumentTemplate(
            category='travel',
            topics=['adventure travel', 'cultural tourism', 'ecotourism', 'business travel', 'luxury travel', 'sustainable tourism', 'medical tourism'],
            sentence_templates=[
                "{topic} offers unique experiences and opportunities for personal growth.",
                "The tourism industry has embraced {topic} as a growing segment.",
                "Sustainable practices are increasingly important in {topic}.",
                "Technology has transformed how people plan and experience {topic}.",
                "Cultural exchange is a key benefit of {topic}.",
                "Safety considerations are essential for successful {topic}.",
                "The economic impact of {topic} on local communities is significant.",
                "Personalized experiences are becoming more common in {topic}.",
                "Global connectivity has made {topic} more accessible.",
                "Future trends in {topic} will focus on sustainability and authenticity."
            ],
            fact_templates=[
                "The global market for {topic} is valued at {number} billion.",
                "Popular destinations for {topic} include {destination1} and {destination2}.",
                "The number of {topic} travelers has increased by {percentage}%.",
                "Key benefits of {topic} include {benefit1} and {benefit2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main benefits of {topic}?",
                "How has technology impacted {topic}?",
                "What are the current trends in {topic}?",
                "How can travelers practice sustainability in {topic}?",
                "What are the top destinations for {topic}?"
            ]
        )
        
        # Food templates
        templates['food'] = DocumentTemplate(
            category='food',
            topics=['nutrition', 'cooking techniques', 'food safety', 'sustainable agriculture', 'food technology', 'culinary arts', 'food culture', 'dietary trends'],
            sentence_templates=[
                "The science of {topic} is essential for health and well-being.",
                "Cultural traditions strongly influence {topic} practices.",
                "Innovation in {topic} has led to new products and techniques.",
                "Education about {topic} is important for public health.",
                "Global cuisine diversity enriches {topic} experiences.",
                "Sustainability is becoming increasingly important in {topic}.",
                "Technology has transformed how we approach {topic}.",
                "Professional expertise in {topic} requires extensive training.",
                "Consumer awareness of {topic} has increased significantly.",
                "Future trends in {topic} will focus on health and sustainability."
            ],
            fact_templates=[
                "The global market for {topic} is valued at {number} billion.",
                "Research in {topic} has shown {percentage}% improvement in outcomes.",
                "Leading experts in {topic} include {expert1} and {expert2}.",
                "Key principles of {topic} include {principle1} and {principle2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the key principles of {topic}?",
                "How has technology impacted {topic}?",
                "What are the current trends in {topic}?",
                "How can individuals improve their {topic}?",
                "What are the health benefits of {topic}?"
            ]
        )
        
        # Psychology templates
        templates['psychology'] = DocumentTemplate(
            category='psychology',
            topics=['cognitive psychology', 'behavioral psychology', 'developmental psychology', 'social psychology', 'clinical psychology', 'neuropsychology', 'positive psychology'],
            sentence_templates=[
                "Research in {topic} has advanced our understanding of human behavior.",
                "Applications of {topic} extend to education, healthcare, and business.",
                "The complexity of {topic} requires rigorous scientific methods.",
                "Cultural factors influence how {topic} manifests across populations.",
                "Technology has opened new avenues for {topic} research and intervention.",
                "Public awareness of {topic} has increased in recent years.",
                "Clinical applications of {topic} help individuals improve their lives.",
                "Theoretical frameworks in {topic} continue to evolve.",
                "Interdisciplinary approaches enhance {topic} research.",
                "Future directions in {topic} will integrate neuroscience and technology."
            ],
            fact_templates=[
                "The first major theory in {topic} was developed in {decade}.",
                "Leading researchers in {topic} include {researcher1} and {researcher2}.",
                "The effectiveness of {topic} interventions is supported by {percentage}% of studies.",
                "Key concepts in {topic} include {concept1} and {concept2}.",
                "Major applications of {topic} include {application1} and {application2}."
            ],
            question_templates=[
                "What are the main theories in {topic}?",
                "How is {topic} applied in practice?",
                "What are the current research trends in {topic}?",
                "How does {topic} impact daily life?",
                "What are the future directions of {topic}?"
            ]
        )
        
        # History templates
        templates['history'] = DocumentTemplate(
            category='history',
            topics=['ancient civilizations', 'medieval history', 'modern history', 'military history', 'cultural history', 'economic history', 'social history'],
            sentence_templates=[
                "The study of {topic} provides insights into human development.",
                "Archaeological discoveries continue to reshape our understanding of {topic}.",
                "The legacy of {topic} influences contemporary society.",
                "Primary sources are essential for studying {topic}.",
                "Interpretations of {topic} have evolved over time.",
                "Global perspectives enrich our understanding of {topic}.",
                "Technology has transformed how we research and teach {topic}.",
                "The lessons of {topic} remain relevant today.",
                "Comparative approaches enhance {topic} scholarship.",
                "Future research in {topic} will integrate new methodologies."
            ],
            fact_templates=[
                "The period of {topic} lasted approximately {number} years.",
                "Key figures in {topic} include {figure1} and {figure2}.",
                "The impact of {topic} is still felt in modern society.",
                "Major events in {topic} include {event1} and {event2}.",
                "Historical debates about {topic} continue among scholars."
            ],
            question_templates=[
                "What are the key events in {topic}?",
                "How does {topic} influence modern society?",
                "What are the main historical debates about {topic}?",
                "How has our understanding of {topic} evolved?",
                "What are the primary sources for studying {topic}?"
            ]
        )
        
        # Philosophy templates
        templates['philosophy'] = DocumentTemplate(
            category='philosophy',
            topics=['ethics', 'metaphysics', 'epistemology', 'political philosophy', 'philosophy of mind', 'philosophy of science', 'existentialism'],
            sentence_templates=[
                "The study of {topic} addresses fundamental questions about existence and knowledge.",
                "Philosophical inquiry into {topic} has evolved over millennia.",
                "Different schools of thought offer diverse perspectives on {topic}.",
                "Contemporary debates in {topic} reflect changing social contexts.",
                "The relevance of {topic} extends to practical decision-making.",
                "Interdisciplinary approaches enrich {topic} scholarship.",
                "Technology presents new challenges for {topic}.",
                "Education in {topic} develops critical thinking skills.",
                "Global perspectives contribute to {topic} discourse.",
                "Future directions in {topic} will address emerging ethical questions."
            ],
            fact_templates=[
                "The first major work on {topic} was written in {period}.",
                "Key philosophers who studied {topic} include {philosopher1} and {philosopher2}.",
                "The influence of {topic} extends to {field1} and {field2}.",
                "Major theories in {topic} include {theory1} and {theory2}.",
                "Contemporary debates in {topic} focus on {debate1} and {debate2}."
            ],
            question_templates=[
                "What are the main theories in {topic}?",
                "How does {topic} apply to modern life?",
                "What are the current debates in {topic}?",
                "Who are the key philosophers in {topic}?",
                "How does {topic} relate to other fields?"
            ]
        )
        
        # Religion templates
        templates['religion'] = DocumentTemplate(
            category='religion',
            topics=['world religions', 'comparative religion', 'religious studies', 'theology', 'religious ethics', 'mysticism', 'religious history'],
            sentence_templates=[
                "The study of {topic} provides insights into human spirituality and culture.",
                "Diverse traditions offer different approaches to {topic}.",
                "Historical context is essential for understanding {topic}.",
                "Interfaith dialogue promotes understanding of {topic}.",
                "The influence of {topic} extends to art, literature, and politics.",
                "Contemporary interpretations of {topic} continue to evolve.",
                "Academic study of {topic} uses rigorous methodologies.",
                "The global diversity of {topic} reflects human cultural richness.",
                "Technology has transformed how people engage with {topic}.",
                "Future research in {topic} will address contemporary challenges."
            ],
            fact_templates=[
                "The origins of {topic} can be traced to {period}.",
                "Major traditions in {topic} include {tradition1} and {tradition2}.",
                "The global population practicing {topic} is approximately {number} billion.",
                "Key texts in {topic} include {text1} and {text2}.",
                "Contemporary issues in {topic} include {issue1} and {issue2}."
            ],
            question_templates=[
                "What are the main traditions in {topic}?",
                "How does {topic} influence society?",
                "What are the contemporary issues in {topic}?",
                "How is {topic} studied academically?",
                "What are the key texts in {topic}?"
            ]
        )
        
        # Politics templates
        templates['politics'] = DocumentTemplate(
            category='politics',
            topics=['comparative politics', 'international relations', 'political theory', 'public policy', 'political economy', 'governance', 'democracy'],
            sentence_templates=[
                "The study of {topic} is essential for understanding power and governance.",
                "Democratic principles influence approaches to {topic}.",
                "Globalization has transformed the landscape of {topic}.",
                "Citizen engagement is crucial for effective {topic}.",
                "Institutional frameworks shape {topic} outcomes.",
                "Technology has created new challenges and opportunities for {topic}.",
                "Policy decisions in {topic} have widespread social impacts.",
                "Comparative analysis enhances understanding of {topic}.",
                "The future of {topic} will be shaped by demographic and technological changes.",
                "Ethical considerations are increasingly important in {topic}."
            ],
            fact_templates=[
                "The first major work on {topic} was published in {year}.",
                "Leading scholars in {topic} include {scholar1} and {scholar2}.",
                "The global population affected by {topic} is approximately {number} billion.",
                "Key institutions in {topic} include {institution1} and {institution2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main theories in {topic}?",
                "How does {topic} affect everyday life?",
                "What are the current challenges in {topic}?",
                "How is {topic} studied?",
                "What are the future trends in {topic}?"
            ]
        )
        
        # Agriculture templates
        templates['agriculture'] = DocumentTemplate(
            category='agriculture',
            topics=['sustainable farming', 'precision agriculture', 'organic farming', 'agritech', 'livestock management', 'crop science', 'agricultural economics'],
            sentence_templates=[
                "Innovation in {topic} is transforming food production systems.",
                "Sustainable practices are increasingly important in {topic}.",
                "Technology has enhanced efficiency and productivity in {topic}.",
                "Climate change presents significant challenges for {topic}.",
                "Global food security depends on advances in {topic}.",
                "Small-scale farmers play a crucial role in {topic}.",
                "Research in {topic} focuses on resilience and sustainability.",
                "Economic factors influence {topic} decisions and practices.",
                "Education and training are essential for modern {topic}.",
                "Future developments in {topic} will address environmental and social concerns."
            ],
            fact_templates=[
                "The global market for {topic} is valued at {number} billion.",
                "Adoption of {topic} technologies has increased by {percentage}%.",
                "Leading countries in {topic} include {country1} and {country2}.",
                "Key benefits of {topic} include {benefit1} and {benefit2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main innovations in {topic}?",
                "How does technology impact {topic}?",
                "What are the sustainability challenges in {topic}?",
                "How does {topic} affect food security?",
                "What are the future trends in {topic}?"
            ]
        )
        
        # Manufacturing templates
        templates['manufacturing'] = DocumentTemplate(
            category='manufacturing',
            topics=['lean manufacturing', 'additive manufacturing', 'industrial automation', 'quality control', 'supply chain management', 'process optimization', 'smart manufacturing'],
            sentence_templates=[
                "The field of {topic} has been revolutionized by digital technology.",
                "Efficiency and quality are key goals in {topic}.",
                "Global competition drives innovation in {topic}.",
                "Sustainability is becoming increasingly important in {topic}.",
                "Automation has transformed {topic} processes.",
                "Data analytics enhances decision-making in {topic}.",
                "Skilled workforce is essential for advanced {topic}.",
                "Supply chain integration is crucial for {topic}.",
                "Future trends in {topic} include Industry 4.0 and IoT integration.",
                "Regulatory compliance affects {topic} operations."
            ],
            fact_templates=[
                "The global manufacturing sector is valued at {number} trillion.",
                "Adoption of {topic} technologies has increased productivity by {percentage}%.",
                "Leading countries in {topic} include {country1} and {country2}.",
                "Key technologies in {topic} include {tech1} and {tech2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main technologies in {topic}?",
                "How has automation impacted {topic}?",
                "What are the sustainability considerations in {topic}?",
                "How does {topic} affect the global economy?",
                "What are the future trends in {topic}?"
            ]
        )
        
        # Energy templates
        templates['energy'] = DocumentTemplate(
            category='energy',
            topics=['renewable energy', 'fossil fuels', 'nuclear energy', 'energy storage', 'smart grids', 'energy efficiency', 'energy policy'],
            sentence_templates=[
                "The transition to {topic} is essential for sustainable development.",
                "Technological innovation is driving advances in {topic}.",
                "Economic factors influence the adoption of {topic}.",
                "Environmental considerations are central to {topic} decisions.",
                "Global energy demand continues to grow, affecting {topic}.",
                "Policy frameworks shape the development of {topic}.",
                "Investment in {topic} infrastructure is increasing globally.",
                "Energy security concerns impact {topic} strategies.",
                "Public awareness of {topic} has increased significantly.",
                "Future energy systems will integrate diverse {topic} sources."
            ],
            fact_templates=[
                "Global investment in {topic} reached {number} billion last year.",
                "The capacity of {topic} has grown by {percentage}% annually.",
                "Leading countries in {topic} include {country1} and {country2}.",
                "Key benefits of {topic} include {benefit1} and {benefit2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main benefits of {topic}?",
                "How does {topic} impact the environment?",
                "What are the current trends in {topic}?",
                "What policies support {topic}?",
                "What are the challenges in {topic} adoption?"
            ]
        )
        
        # Transportation templates
        templates['transportation'] = DocumentTemplate(
            category='transportation',
            topics=['public transit', 'electric vehicles', 'autonomous vehicles', 'aviation', 'maritime transport', 'logistics', 'sustainable transportation'],
            sentence_templates=[
                "Innovation in {topic} is transforming how people and goods move.",
                "Sustainability is a key consideration in modern {topic}.",
                "Technology has enhanced safety and efficiency in {topic}.",
                "Urban planning is closely linked to {topic} systems.",
                "Global trade depends on efficient {topic} networks.",
                "Public investment in {topic} infrastructure is crucial.",
                "Consumer preferences are shaping the future of {topic}.",
                "Regulatory frameworks govern {topic} operations.",
                "Integration of different {topic} modes improves efficiency.",
                "Future trends in {topic} include automation and electrification."
            ],
            fact_templates=[
                "The global market for {topic} is valued at {number} billion.",
                "Adoption of {topic} technologies has increased by {percentage}%.",
                "Leading companies in {topic} include {company1} and {company2}.",
                "Key benefits of {topic} include {benefit1} and {benefit2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main innovations in {topic}?",
                "How does {topic} impact the environment?",
                "What are the current trends in {topic}?",
                "How is {topic} regulated?",
                "What are the future directions of {topic}?"
            ]
        )
        
        # Real Estate templates
        templates['real_estate'] = DocumentTemplate(
            category='real_estate',
            topics=['residential real estate', 'commercial real estate', 'property management', 'real estate investment', 'sustainable building', 'smart homes', 'urban development'],
            sentence_templates=[
                "The market for {topic} is influenced by economic and demographic factors.",
                "Technology is transforming how {topic} is bought, sold, and managed.",
                "Sustainability is becoming increasingly important in {topic}.",
                "Investment in {topic} requires careful analysis and planning.",
                "Urban development trends affect {topic} values and demand.",
                "Regulatory frameworks govern {topic} transactions and development.",
                "Professional expertise is essential for successful {topic} operations.",
                "Global economic conditions impact {topic} markets.",
                "Consumer preferences shape the future of {topic}.",
                "Innovation in {topic} includes smart technology and green building."
            ],
            fact_templates=[
                "The global real estate market is valued at {number} trillion.",
                "Investment in {topic} has grown by {percentage}% annually.",
                "Leading markets for {topic} include {market1} and {market2}.",
                "Key factors affecting {topic} include {factor1} and {factor2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main factors affecting {topic}?",
                "How has technology impacted {topic}?",
                "What are the current trends in {topic}?",
                "How is {topic} regulated?",
                "What are the investment considerations in {topic}?"
            ]
        )
        
        # Retail templates
        templates['retail'] = DocumentTemplate(
            category='retail',
            topics=['e-commerce', 'brick and mortar', 'omnichannel retail', 'retail technology', 'customer experience', 'supply chain', 'retail analytics'],
            sentence_templates=[
                "The retail industry has been transformed by digital technology.",
                "Consumer behavior continues to evolve, affecting {topic}.",
                "Omnichannel strategies are essential for modern {topic}.",
                "Data analytics drives decision-making in {topic}.",
                "Customer experience is a key differentiator in {topic}.",
                "Sustainability is becoming important in {topic} practices.",
                "Global supply chains impact {topic} operations.",
                "Innovation in {topic} includes AI, AR, and personalization.",
                "Economic factors influence {topic} performance.",
                "Future trends in {topic} will focus on integration and experience."
            ],
            fact_templates=[
                "The global retail market is valued at {number} trillion.",
                "E-commerce sales in {topic} have grown by {percentage}%.",
                "Leading retailers in {topic} include {retailer1} and {retailer2}.",
                "Key technologies in {topic} include {tech1} and {tech2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "How has technology transformed {topic}?",
                "What are the current trends in {topic}?",
                "How does {topic} impact consumer behavior?",
                "What are the key success factors in {topic}?",
                "What are the future directions of {topic}?"
            ]
        )
        
        # Hospitality templates
        templates['hospitality'] = DocumentTemplate(
            category='hospitality',
            topics=['hotel management', 'restaurant management', 'tourism', 'event planning', 'customer service', 'sustainable hospitality', 'hospitality technology'],
            sentence_templates=[
                "The {topic} industry focuses on creating exceptional guest experiences.",
                "Technology has transformed operations in {topic}.",
                "Sustainability is increasingly important in {topic}.",
                "Cultural sensitivity is essential for successful {topic}.",
                "Economic conditions affect {topic} demand and performance.",
                "Professional training is crucial for {topic} careers.",
                "Global tourism trends influence {topic} strategies.",
                "Innovation in {topic} includes personalization and automation.",
                "Customer reviews impact {topic} reputation and success.",
                "Future trends in {topic} will emphasize experience and sustainability."
            ],
            fact_templates=[
                "The global {topic} industry is valued at {number} billion.",
                "Employment in {topic} has grown by {percentage}%.",
                "Leading companies in {topic} include {company1} and {company2}.",
                "Key markets for {topic} include {market1} and {market2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the key success factors in {topic}?",
                "How has technology impacted {topic}?",
                "What are the current trends in {topic}?",
                "How does {topic} contribute to the economy?",
                "What are the future directions of {topic}?"
            ]
        )
        
        # Consulting templates
        templates['consulting'] = DocumentTemplate(
            category='consulting',
            topics=['management consulting', 'strategy consulting', 'IT consulting', 'financial consulting', 'HR consulting', 'sustainability consulting'],
            sentence_templates=[
                "{topic} helps organizations improve performance and solve complex problems.",
                "Data analytics is increasingly important in {topic}.",
                "Industry expertise is essential for effective {topic}.",
                "Client relationships are crucial in {topic} success.",
                "Technology has transformed {topic} delivery and capabilities.",
                "Globalization has expanded the scope of {topic}.",
                "Ethical considerations are important in {topic} practice.",
                "Innovation in {topic} includes new methodologies and tools.",
                "Economic conditions affect {topic} demand.",
                "Future trends in {topic} will focus on digital transformation and sustainability."
            ],
            fact_templates=[
                "The global {topic} market is valued at {number} billion.",
                "Leading firms in {topic} include {firm1} and {firm2}.",
                "The average project in {topic} lasts {number} months.",
                "Key services in {topic} include {service1} and {service2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main services in {topic}?",
                "How has technology impacted {topic}?",
                "What are the current trends in {topic}?",
                "What skills are needed for {topic}?",
                "How does {topic} create value for clients?"
            ]
        )
        
        # Nonprofit templates
        templates['nonprofit'] = DocumentTemplate(
            category='nonprofit',
            topics=['nonprofit management', 'fundraising', 'social entrepreneurship', 'volunteer management', 'nonprofit finance', 'advocacy', 'program evaluation'],
            sentence_templates=[
                "{topic} organizations address social and environmental challenges.",
                "Funding is a critical concern for {topic} sustainability.",
                "Volunteer engagement is essential for {topic} operations.",
                "Impact measurement is increasingly important in {topic}.",
                "Collaboration enhances {topic} effectiveness.",
                "Technology has transformed how {topic} organizations operate.",
                "Public awareness supports {topic} missions.",
                "Professional management is crucial for {topic} success.",
                "Policy advocacy is a key function of {topic}.",
                "Future trends in {topic} include social innovation and digital engagement."
            ],
            fact_templates=[
                "The global {topic} sector employs {number} million people.",
                "Donations to {topic} have grown by {percentage}% annually.",
                "Leading {topic} organizations include {org1} and {org2}.",
                "Key challenges in {topic} include {challenge1} and {challenge2}.",
                "Major areas of {topic} work include {area1} and {area2}."
            ],
            question_templates=[
                "What are the main challenges in {topic}?",
                "How does {topic} create social impact?",
                "What are the current trends in {topic}?",
                "How can individuals support {topic}?",
                "What are the future directions of {topic}?"
            ]
        )
        
        # Government templates
        templates['government'] = DocumentTemplate(
            category='government',
            topics=['public administration', 'policy making', 'digital government', 'public finance', 'regulatory affairs', 'municipal government', 'federal government'],
            sentence_templates=[
                "{topic} serves the public interest through various programs and services.",
                "Efficiency and transparency are important in {topic} operations.",
                "Technology has transformed how {topic} delivers services.",
                "Citizen engagement is essential for effective {topic}.",
                "Budget constraints affect {topic} priorities and operations.",
                "Policy decisions in {topic} have widespread social impacts.",
                "Professional civil service is crucial for {topic} effectiveness.",
                "Interagency coordination improves {topic} outcomes.",
                "Future trends in {topic} include digital transformation and data-driven decision-making.",
                "Ethical standards guide {topic} practices and decisions."
            ],
            fact_templates=[
                "The {topic} budget is approximately {number} billion.",
                "The number of {topic} employees is approximately {number} thousand.",
                "Key departments in {topic} include {dept1} and {dept2}.",
                "Major initiatives in {topic} include {initiative1} and {initiative2}.",
                "Public satisfaction with {topic} is at {percentage}%."
            ],
            question_templates=[
                "What are the main functions of {topic}?",
                "How has technology impacted {topic}?",
                "What are the current challenges in {topic}?",
                "How does {topic} serve the public?",
                "What are the future trends in {topic}?"
            ]
        )
        
        # Communication templates
        templates['communication'] = DocumentTemplate(
            category='communication',
            topics=['public relations', 'corporate communication', 'internal communication', 'crisis communication', 'strategic communication', 'digital communication'],
            sentence_templates=[
                "Effective {topic} is essential for organizational success.",
                "Technology has transformed how {topic} is practiced.",
                "Stakeholder engagement is crucial in {topic}.",
                "Crisis management is a key aspect of {topic}.",
                "Measurement and evaluation improve {topic} effectiveness.",
                "Global perspectives influence {topic} strategies.",
                "Ethical considerations guide {topic} practices.",
                "Integration across channels enhances {topic} impact.",
                "Future trends in {topic} include personalization and interactivity.",
                "Professional expertise is essential for strategic {topic}."
            ],
            fact_templates=[
                "The global {topic} market is valued at {number} billion.",
                "Leading agencies in {topic} include {agency1} and {agency2}.",
                "The use of digital channels in {topic} has grown by {percentage}%.",
                "Key metrics in {topic} include {metric1} and {metric2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the key principles of {topic}?",
                "How has technology impacted {topic}?",
                "What are the current trends in {topic}?",
                "How is {topic} measured and evaluated?",
                "What are the future directions of {topic}?"
            ]
        )
        
        # Design templates
        templates['design'] = DocumentTemplate(
            category='design',
            topics=['graphic design', 'industrial design', 'UX design', 'interior design', 'fashion design', 'web design', 'design thinking'],
            sentence_templates=[
                "The field of {topic} combines creativity with functionality.",
                "Technology has expanded the possibilities for {topic}.",
                "User-centered approaches are essential in modern {topic}.",
                "Sustainability is increasingly important in {topic}.",
                "Cultural influences shape {topic} trends and practices.",
                "Collaboration enhances {topic} outcomes.",
                "Professional training is crucial for {topic} careers.",
                "Innovation in {topic} includes new tools and methodologies.",
                "Global perspectives enrich {topic} practice.",
                "Future trends in {topic} will focus on sustainability and technology integration."
            ],
            fact_templates=[
                "The global {topic} industry is valued at {number} billion.",
                "Employment in {topic} has grown by {percentage}%.",
                "Leading firms in {topic} include {firm1} and {firm2}.",
                "Key tools in {topic} include {tool1} and {tool2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the key principles of {topic}?",
                "How has technology impacted {topic}?",
                "What are the current trends in {topic}?",
                "What skills are needed for {topic}?",
                "What are the future directions of {topic}?"
            ]
        )
        
        # Innovation templates
        templates['innovation'] = DocumentTemplate(
            category='innovation',
            topics=['product innovation', 'process innovation', 'business model innovation', 'social innovation', 'open innovation', 'disruptive innovation'],
            sentence_templates=[
                "{topic} drives economic growth and competitive advantage.",
                "Technology has accelerated the pace of {topic}.",
                "Collaboration enhances {topic} outcomes.",
                "Risk-taking is inherent in {topic} processes.",
                "Market needs inspire successful {topic}.",
                "Sustainability considerations guide modern {topic}.",
                "Measurement and evaluation improve {topic} effectiveness.",
                "Global competition drives {topic} investment.",
                "Future trends in {topic} include AI and sustainability focus.",
                "Organizational culture impacts {topic} success."
            ],
            fact_templates=[
                "Global investment in {topic} reached {number} billion last year.",
                "The rate of {topic} has increased by {percentage}%.",
                "Leading companies in {topic} include {company1} and {company2}.",
                "Key drivers of {topic} include {driver1} and {driver2}.",
                "Major barriers to {topic} include {barrier1} and {barrier2}."
            ],
            question_templates=[
                "What are the main types of {topic}?",
                "How can organizations foster {topic}?",
                "What are the current trends in {topic}?",
                "How is {topic} measured?",
                "What are the challenges in {topic}?"
            ]
        )
        
        # Leadership templates
        templates['leadership'] = DocumentTemplate(
            category='leadership',
            topics=['executive leadership', 'team leadership', 'organizational leadership', 'transformational leadership', 'servant leadership', 'adaptive leadership'],
            sentence_templates=[
                "Effective {topic} is essential for organizational success.",
                "Different situations require different {topic} styles.",
                "Emotional intelligence is crucial for {topic}.",
                "Communication skills are fundamental to {topic}.",
                "Continuous learning is important for {topic} development.",
                "Ethical considerations guide {topic} decisions.",
                "Technology has transformed how {topic} is practiced.",
                "Global perspectives influence {topic} approaches.",
                "Future trends in {topic} emphasize adaptability and inclusion.",
                "Mentorship and coaching support {topic} development."
            ],
            fact_templates=[
                "Organizations with strong {topic} perform {percentage}% better.",
                "Investment in {topic} development has increased by {percentage}%.",
                "Key skills for {topic} include {skill1} and {skill2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}.",
                "Effective {topic} can improve employee engagement by {percentage}%."
            ],
            question_templates=[
                "What are the key styles of {topic}?",
                "How can individuals develop {topic} skills?",
                "What are the current trends in {topic}?",
                "How does {topic} impact organizational performance?",
                "What are the challenges in {topic}?"
            ]
        )
        
        # Project Management templates
        templates['project_management'] = DocumentTemplate(
            category='project_management',
            topics=['agile project management', 'waterfall project management', 'project planning', 'risk management', 'stakeholder management', 'project governance'],
            sentence_templates=[
                "Effective {topic} is essential for project success.",
                "Methodologies in {topic} include agile and waterfall approaches.",
                "Risk management is a key aspect of {topic}.",
                "Stakeholder engagement is crucial for {topic}.",
                "Technology has enhanced {topic} tools and processes.",
                "Communication is fundamental to successful {topic}.",
                "Resource allocation affects {topic} outcomes.",
                "Monitoring and control are essential in {topic}.",
                "Future trends in {topic} include AI integration and remote collaboration.",
                "Professional certification is valued in {topic}."
            ],
            fact_templates=[
                "Organizations using {topic} report {percentage}% higher success rates.",
                "The global market for {topic} tools is valued at {number} billion.",
                "Key methodologies in {topic} include {method1} and {method2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}.",
                "Effective {topic} can reduce project overruns by {percentage}%."
            ],
            question_templates=[
                "What are the main methodologies in {topic}?",
                "How does {topic} improve project success?",
                "What are the current trends in {topic}?",
                "What tools are used in {topic}?",
                "What are the challenges in {topic}?"
            ]
        )
        
        # Data Science templates
        templates['data_science'] = DocumentTemplate(
            category='data_science',
            topics=['machine learning', 'data visualization', 'big data analytics', 'predictive analytics', 'data mining', 'data engineering', 'AI ethics'],
            sentence_templates=[
                "{topic} has transformed how organizations make decisions.",
                "Data quality is fundamental to successful {topic}.",
                "Technology has enabled advanced {topic} techniques.",
                "Ethical considerations are important in {topic} practice.",
                "Business value drives {topic} investments.",
                "Interdisciplinary approaches enhance {topic} outcomes.",
                "Real-time analytics is becoming important in {topic}.",
                "Cloud computing supports scalable {topic} infrastructure.",
                "Future trends in {topic} include automated ML and AI integration.",
                "Data privacy concerns affect {topic} practices."
            ],
            fact_templates=[
                "The global market for {topic} is valued at {number} billion.",
                "Organizations using {topic} report {percentage}% better decisions.",
                "Key tools in {topic} include {tool1} and {tool2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}.",
                "The demand for {topic} professionals has grown by {percentage}%."
            ],
            question_templates=[
                "What are the main techniques in {topic}?",
                "How does {topic} create business value?",
                "What are the current trends in {topic}?",
                "What tools are used in {topic}?",
                "What are the ethical considerations in {topic}?"
            ]
        )
        
        # Cybersecurity templates
        templates['cybersecurity'] = DocumentTemplate(
            category='cybersecurity',
            topics=['network security', 'application security', 'cloud security', 'identity management', 'threat intelligence', 'incident response', 'security governance'],
            sentence_templates=[
                "{topic} is essential for protecting digital assets and information.",
                "Cyber threats continue to evolve, challenging {topic}.",
                "Technology advances both {topic} solutions and threats.",
                "Compliance requirements drive {topic} investments.",
                "Human factors are critical in {topic} effectiveness.",
                "Risk management is fundamental to {topic}.",
                "Incident response is a key component of {topic}.",
                "Collaboration enhances {topic} defense.",
                "Future trends in {topic} include AI-powered security and zero trust.",
                "Professional certification is valued in {topic}."
            ],
            fact_templates=[
                "Global cybercrime damages reached {number} billion last year.",
                "Organizations investing in {topic} reduced breaches by {percentage}%.",
                "Key frameworks in {topic} include {framework1} and {framework2}.",
                "Major threats in {topic} include {threat1} and {threat2}.",
                "The demand for {topic} professionals has grown by {percentage}%."
            ],
            question_templates=[
                "What are the main threats addressed by {topic}?",
                "How can organizations improve {topic}?",
                "What are the current trends in {topic}?",
                "What frameworks guide {topic}?",
                "What are the challenges in {topic}?"
            ]
        )
        
        # Blockchain templates
        templates['blockchain'] = DocumentTemplate(
            category='blockchain',
            topics=['cryptocurrency', 'smart contracts', 'decentralized finance', 'NFTs', 'enterprise blockchain', 'blockchain governance', 'token economics'],
            sentence_templates=[
                "{topic} has the potential to transform various industries.",
                "Decentralization is a key feature of {topic}.",
                "Technology continues to evolve in {topic} applications.",
                "Regulatory frameworks are developing for {topic}.",
                "Investment in {topic} has grown significantly.",
                "Security considerations are important in {topic}.",
                "Interoperability is a challenge for {topic} adoption.",
                "Scalability solutions are being developed for {topic}.",
                "Future trends in {topic} include institutional adoption and integration.",
                "Education about {topic} is essential for widespread adoption."
            ],
            fact_templates=[
                "The market capitalization of {topic} reached {number} billion.",
                "Adoption of {topic} has grown by {percentage}% annually.",
                "Leading platforms in {topic} include {platform1} and {platform2}.",
                "Key applications of {topic} include {app1} and {app2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main applications of {topic}?",
                "How does {topic} work?",
                "What are the current trends in {topic}?",
                "What are the regulatory considerations for {topic}?",
                "What are the challenges in {topic} adoption?"
            ]
        )
        
        # IoT templates
        templates['iot'] = DocumentTemplate(
            category='iot',
            topics=['smart home', 'industrial IoT', 'IoT security', 'IoT analytics', 'edge computing', 'IoT platforms', 'connected devices'],
            sentence_templates=[
                "{topic} is transforming how devices interact and communicate.",
                "Connectivity is fundamental to {topic} functionality.",
                "Security is a critical concern in {topic} deployment.",
                "Data analytics enhances {topic} value.",
                "Technology advances drive {topic} innovation.",
                "Standardization challenges affect {topic} adoption.",
                "Energy efficiency is important for {topic} devices.",
                "Integration with other systems enhances {topic} utility.",
                "Future trends in {topic} include AI integration and 5G.",
                "Privacy concerns affect {topic} acceptance."
            ],
            fact_templates=[
                "The number of {topic} devices is expected to reach {number} billion by {year}.",
                "The market for {topic} is valued at {number} billion.",
                "Leading companies in {topic} include {company1} and {company2}.",
                "Key applications of {topic} include {app1} and {app2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main applications of {topic}?",
                "How does {topic} work?",
                "What are the security considerations for {topic}?",
                "What are the current trends in {topic}?",
                "What are the challenges in {topic} adoption?"
            ]
        )
        
        # 5G templates
        templates['5g'] = DocumentTemplate(
            category='5g',
            topics=['5G networks', '5G applications', '5G infrastructure', '5G security', '5G in healthcare', '5G in manufacturing', '5G in smart cities'],
            sentence_templates=[
                "{topic} represents a major advancement in wireless communication.",
                "Speed and latency improvements define {topic} capabilities.",
                "Infrastructure investment is crucial for {topic} deployment.",
                "Security considerations are important in {topic} implementation.",
                "Applications of {topic} span multiple industries.",
                "Technology continues to evolve within {topic} standards.",
                "Global competition drives {topic} development.",
                "Economic impact of {topic} is significant.",
                "Future trends in {topic} include 6G research and integration.",
                "Consumer adoption of {topic} is growing steadily."
            ],
            fact_templates=[
                "Global investment in {topic} reached {number} billion last year.",
                "The number of {topic} connections is expected to reach {number} billion by {year}.",
                "Leading countries in {topic} include {country1} and {country2}.",
                "Key benefits of {topic} include {benefit1} and {benefit2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main benefits of {topic}?",
                "How does {topic} differ from previous generations?",
                "What are the applications of {topic}?",
                "What are the infrastructure requirements for {topic}?",
                "What are the future trends in {topic}?"
            ]
        )
        
        # Quantum Computing templates
        templates['quantum_computing'] = DocumentTemplate(
            category='quantum_computing',
            topics=['quantum algorithms', 'quantum hardware', 'quantum cryptography', 'quantum simulation', 'quantum error correction', 'quantum supremacy'],
            sentence_templates=[
                "{topic} represents a paradigm shift in computational power.",
                "Quantum mechanics principles enable {topic} functionality.",
                "Research in {topic} is advancing rapidly.",
                "Applications of {topic} could revolutionize various fields.",
                "Technical challenges remain in {topic} development.",
                "Investment in {topic} has increased significantly.",
                "Collaboration between academia and industry drives {topic} progress.",
                "Security implications of {topic} are significant.",
                "Future breakthroughs in {topic} could enable new applications.",
                "Education about {topic} is essential for workforce preparation."
            ],
            fact_templates=[
                "Global investment in {topic} reached {number} billion last year.",
                "The number of qubits in {topic} systems has grown by {percentage}%.",
                "Leading companies in {topic} include {company1} and {company2}.",
                "Key applications of {topic} include {app1} and {app2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main applications of {topic}?",
                "How does {topic} work?",
                "What are the current challenges in {topic}?",
                "What are the security implications of {topic}?",
                "What are the future directions of {topic}?"
            ]
        )
        
        # Robotics templates
        templates['robotics'] = DocumentTemplate(
            category='robotics',
            topics=['industrial robots', 'service robots', 'medical robots', 'autonomous robots', 'robotics software', 'robotics ethics', 'human-robot interaction'],
            sentence_templates=[
                "{topic} is transforming automation across industries.",
                "Advances in AI enhance {topic} capabilities.",
                "Safety is a critical consideration in {topic} deployment.",
                "Human-robot collaboration is a growing trend in {topic}.",
                "Cost reduction drives {topic} adoption.",
                "Technology continues to advance {topic} functionality.",
                "Applications of {topic} span multiple sectors.",
                "Ethical considerations guide {topic} development.",
                "Future trends in {topic} include increased autonomy and AI integration.",
                "Workforce implications of {topic} require attention."
            ],
            fact_templates=[
                "The global market for {topic} is valued at {number} billion.",
                "The number of {topic} units sold has grown by {percentage}%.",
                "Leading companies in {topic} include {company1} and {company2}.",
                "Key applications of {topic} include {app1} and {app2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main applications of {topic}?",
                "How has AI impacted {topic}?",
                "What are the safety considerations for {topic}?",
                "What are the current trends in {topic}?",
                "What are the ethical considerations in {topic}?"
            ]
        )
        
        # Biotechnology templates
        templates['biotechnology'] = DocumentTemplate(
            category='biotechnology',
            topics=['genetic engineering', 'pharmaceutical biotechnology', 'agricultural biotechnology', 'industrial biotechnology', 'biotech ethics', 'regenerative medicine', 'synthetic biology'],
            sentence_templates=[
                "{topic} has revolutionized medicine, agriculture, and industry.",
                "Genetic engineering techniques enable {topic} applications.",
                "Regulatory frameworks govern {topic} development and deployment.",
                "Investment in {topic} has grown significantly.",
                "Ethical considerations are important in {topic} research.",
                "Technology advances drive {topic} innovation.",
                "Applications of {topic} address global challenges.",
                "Public awareness of {topic} has increased.",
                "Future trends in {topic} include personalized medicine and sustainable solutions.",
                "Collaboration accelerates {topic} progress."
            ],
            fact_templates=[
                "The global market for {topic} is valued at {number} billion.",
                "Investment in {topic} has grown by {percentage}% annually.",
                "Leading companies in {topic} include {company1} and {company2}.",
                "Key applications of {topic} include {app1} and {app2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main applications of {topic}?",
                "How does {topic} work?",
                "What are the ethical considerations in {topic}?",
                "What are the current trends in {topic}?",
                "What are the regulatory aspects of {topic}?"
            ]
        )
        
        # Nanotechnology templates
        templates['nanotechnology'] = DocumentTemplate(
            category='nanotechnology',
            topics=['nanomaterials', 'nanomedicine', 'nanoelectronics', 'nanofabrication', 'nanotoxicology', 'nanosensors', 'nanorobotics'],
            sentence_templates=[
                "{topic} manipulates matter at the atomic and molecular scale.",
                "Applications of {topic} span multiple industries.",
                "Safety considerations are important in {topic} development.",
                "Technology advances enable new {topic} applications.",
                "Investment in {topic} has increased significantly.",
                "Interdisciplinary research drives {topic} innovation.",
                "Standardization challenges affect {topic} adoption.",
                "Future trends in {topic} include advanced materials and applications.",
                "Public awareness of {topic} is growing.",
                "Regulatory frameworks are developing for {topic}."
            ],
            fact_templates=[
                "The global market for {topic} is valued at {number} billion.",
                "Investment in {topic} has grown by {percentage}% annually.",
                "Leading applications of {topic} include {app1} and {app2}.",
                "Key materials in {topic} include {material1} and {material2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main applications of {topic}?",
                "How does {topic} work?",
                "What are the safety considerations for {topic}?",
                "What are the current trends in {topic}?",
                "What are the regulatory aspects of {topic}?"
            ]
        )
        
        # Space Technology templates
        templates['space_technology'] = DocumentTemplate(
            category='space_technology',
            topics=['satellites', 'space exploration', 'space tourism', 'asteroid mining', 'space manufacturing', 'space habitats', 'space debris'],
            sentence_templates=[
                "{topic} has expanded human presence beyond Earth.",
                "Technology advances enable new {topic} capabilities.",
                "Commercial investment is growing in {topic}.",
                "International cooperation supports {topic} development.",
                "Safety is paramount in {topic} operations.",
                "Sustainability considerations affect {topic} planning.",
                "Applications of {topic} benefit life on Earth.",
                "Future trends in {topic} include Mars missions and space stations.",
                "Regulatory frameworks govern {topic} activities.",
                "Public interest in {topic} remains high."
            ],
            fact_templates=[
                "Global investment in {topic} reached {number} billion last year.",
                "The number of {topic} launches has grown by {percentage}%.",
                "Leading countries in {topic} include {country1} and {country2}.",
                "Key applications of {topic} include {app1} and {app2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main applications of {topic}?",
                "How has commercialization affected {topic}?",
                "What are the safety considerations for {topic}?",
                "What are the current trends in {topic}?",
                "What are the future directions of {topic}?"
            ]
        )
        
        # Virtual Reality templates
        templates['virtual_reality'] = DocumentTemplate(
            category='virtual_reality',
            topics=['VR gaming', 'VR education', 'VR training', 'VR therapy', 'VR social', 'VR enterprise', 'VR hardware'],
            sentence_templates=[
                "{topic} creates immersive digital experiences.",
                "Technology advances enhance {topic} capabilities.",
                "Applications of {topic} span multiple sectors.",
                "User experience is crucial for {topic} adoption.",
                "Content development drives {topic} growth.",
                "Health considerations affect {topic} use.",
                "Social applications are expanding in {topic}.",
                "Future trends in {topic} include improved hardware and applications.",
                "Investment in {topic} has increased significantly.",
                "Accessibility is important for {topic} adoption."
            ],
            fact_templates=[
                "The global market for {topic} is valued at {number} billion.",
                "The number of {topic} users has grown by {percentage}%.",
                "Leading companies in {topic} include {company1} and {company2}.",
                "Key applications of {topic} include {app1} and {app2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main applications of {topic}?",
                "How does {topic} work?",
                "What are the health considerations for {topic}?",
                "What are the current trends in {topic}?",
                "What are the future directions of {topic}?"
            ]
        )
        
        # Augmented Reality templates
        templates['augmented_reality'] = DocumentTemplate(
            category='augmented_reality',
            topics=['AR gaming', 'AR retail', 'AR education', 'AR industrial', 'AR navigation', 'AR healthcare', 'AR social'],
            sentence_templates=[
                "{topic} overlays digital content on the real world.",
                "Mobile devices enable widespread {topic} adoption.",
                "Applications of {topic} enhance productivity and experience.",
                "Technology advances improve {topic} capabilities.",
                "Integration with other systems enhances {topic} utility.",
                "Privacy considerations affect {topic} deployment.",
                "Enterprise adoption of {topic} is growing.",
                "Future trends in {topic} include improved hardware and applications.",
                "Content development is crucial for {topic} success.",
                "User experience determines {topic} adoption."
            ],
            fact_templates=[
                "The global market for {topic} is valued at {number} billion.",
                "The number of {topic} applications has grown by {percentage}%.",
                "Leading platforms in {topic} include {platform1} and {platform2}.",
                "Key applications of {topic} include {app1} and {app2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main applications of {topic}?",
                "How does {topic} differ from VR?",
                "What are the current trends in {topic}?",
                "What are the privacy considerations for {topic}?",
                "What are the future directions of {topic}?"
            ]
        )
        
        # 3D Printing templates
        templates['3d_printing'] = DocumentTemplate(
            category='3d_printing',
            topics=['industrial 3D printing', 'medical 3D printing', 'consumer 3D printing', '3D printing materials', '3D printing services', '3D printing software'],
            sentence_templates=[
                "{topic} enables additive manufacturing of complex objects.",
                "Technology advances expand {topic} capabilities.",
                "Applications of {topic} span multiple industries.",
                "Material innovation drives {topic} progress.",
                "Cost reduction enables wider {topic} adoption.",
                "Quality control is important in {topic} production.",
                "Sustainability benefits include reduced waste in {topic}.",
                "Future trends in {topic} include larger scale and new materials.",
                "Regulatory considerations affect {topic} in certain applications.",
                "Education about {topic} is essential for adoption."
            ],
            fact_templates=[
                "The global market for {topic} is valued at {number} billion.",
                "The number of {topic} units sold has grown by {percentage}%.",
                "Leading companies in {topic} include {company1} and {company2}.",
                "Key materials in {topic} include {material1} and {material2}.",
                "Major applications of {topic} include {app1} and {app2}."
            ],
            question_templates=[
                "What are the main applications of {topic}?",
                "How does {topic} work?",
                "What are the current trends in {topic}?",
                "What are the sustainability benefits of {topic}?",
                "What are the future directions of {topic}?"
            ]
        )
        
        # Drones templates
        templates['drones'] = DocumentTemplate(
            category='drones',
            topics=['commercial drones', 'consumer drones', 'military drones', 'drone delivery', 'drone photography', 'drone regulations', 'drone technology'],
            sentence_templates=[
                "{topic} have transformed aerial capabilities across industries.",
                "Technology advances enhance {topic} performance and safety.",
                "Applications of {topic} include delivery, photography, and inspection.",
                "Regulatory frameworks govern {topic} operations.",
                "Privacy concerns affect {topic} deployment.",
                "Autonomous capabilities are developing in {topic}.",
                "Investment in {topic} has increased significantly.",
                "Future trends in {topic} include AI integration and new applications.",
                "Safety is paramount in {topic} operations.",
                "Public acceptance of {topic} varies by application."
            ],
            fact_templates=[
                "The global market for {topic} is valued at {number} billion.",
                "The number of {topic} units sold has grown by {percentage}%.",
                "Leading companies in {topic} include {company1} and {company2}.",
                "Key applications of {topic} include {app1} and {app2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main applications of {topic}?",
                "How does {topic} technology work?",
                "What are the regulatory considerations for {topic}?",
                "What are the current trends in {topic}?",
                "What are the future directions of {topic}?"
            ]
        )
        
        # Smart Cities templates
        templates['smart_cities'] = DocumentTemplate(
            category='smart_cities',
            topics=['smart infrastructure', 'smart transportation', 'smart energy', 'smart governance', 'smart buildings', 'smart waste management', 'citizen engagement'],
            sentence_templates=[
                "{topic} use technology to improve urban life quality.",
                "IoT sensors enable {topic} functionality.",
                "Data analytics drives {topic} decision-making.",
                "Sustainability is a key goal of {topic} initiatives.",
                "Citizen engagement is important for {topic} success.",
                "Investment in {topic} infrastructure is growing globally.",
                "Integration across systems enhances {topic} effectiveness.",
                "Privacy considerations affect {topic} data collection.",
                "Future trends in {topic} include AI and 5G integration.",
                "Equity considerations should guide {topic} development."
            ],
            fact_templates=[
                "Global investment in {topic} reached {number} billion last year.",
                "The number of {topic} projects has grown by {percentage}%.",
                "Leading cities in {topic} include {city1} and {city2}.",
                "Key technologies in {topic} include {tech1} and {tech2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main components of {topic}?",
                "How does {topic} improve urban life?",
                "What are the current trends in {topic}?",
                "What are the privacy considerations for {topic}?",
                "What are the future directions of {topic}?"
            ]
        )
        
        # Digital Health templates
        templates['digital_health'] = DocumentTemplate(
            category='digital_health',
            topics=['telemedicine', 'health apps', 'wearable devices', 'electronic health records', 'health AI', 'digital therapeutics', 'remote monitoring'],
            sentence_templates=[
                "{topic} has transformed healthcare delivery and access.",
                "Technology enables new {topic} capabilities.",
                "Patient engagement is enhanced by {topic} solutions.",
                "Regulatory frameworks govern {topic} applications.",
                "Data security is crucial for {topic} adoption.",
                "Integration with healthcare systems is important for {topic}.",
                "Evidence-based validation is needed for {topic} effectiveness.",
                "Future trends in {topic} include AI and personalized medicine.",
                "Accessibility is improved through {topic} solutions.",
                "Cost considerations affect {topic} implementation."
            ],
            fact_templates=[
                "The global market for {topic} is valued at {number} billion.",
                "The number of {topic} users has grown by {percentage}%.",
                "Leading companies in {topic} include {company1} and {company2}.",
                "Key applications of {topic} include {app1} and {app2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main applications of {topic}?",
                "How does {topic} improve healthcare?",
                "What are the regulatory considerations for {topic}?",
                "What are the current trends in {topic}?",
                "What are the future directions of {topic}?"
            ]
        )
        
        # Fintech templates
        templates['fintech'] = DocumentTemplate(
            category='fintech',
            topics=['digital payments', 'peer-to-peer lending', 'robo-advisors', 'insurtech', 'regtech', 'neobanks', 'blockchain finance'],
            sentence_templates=[
                "{topic} has transformed financial services delivery.",
                "Technology enables new {topic} business models.",
                "Regulatory frameworks are adapting to {topic} innovation.",
                "Consumer adoption of {topic} has grown significantly.",
                "Traditional finance institutions are embracing {topic}.",
                "Security is paramount in {topic} applications.",
                "Data analytics drives {topic} decision-making.",
                "Global expansion is a trend in {topic} development.",
                "Future trends in {topic} include AI and embedded finance.",
                "Financial inclusion is improved through {topic} solutions."
            ],
            fact_templates=[
                "The global market for {topic} is valued at {number} billion.",
                "Investment in {topic} has grown by {percentage}% annually.",
                "Leading companies in {topic} include {company1} and {company2}.",
                "Key services in {topic} include {service1} and {service2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main services in {topic}?",
                "How has {topic} transformed finance?",
                "What are the regulatory considerations for {topic}?",
                "What are the current trends in {topic}?",
                "What are the future directions of {topic}?"
            ]
        )
        
        # EdTech templates
        templates['edtech'] = DocumentTemplate(
            category='edtech',
            topics=['online learning platforms', 'educational apps', 'learning management systems', 'adaptive learning', 'gamification', 'virtual classrooms', 'AI in education'],
            sentence_templates=[
                "{topic} has transformed how education is delivered and accessed.",
                "Technology enables personalized {topic} experiences.",
                "Accessibility is improved through {topic} solutions.",
                "Data analytics enhances {topic} effectiveness.",
                "Teacher training is important for {topic} adoption.",
                "Integration with traditional education is key for {topic} success.",
                "Equity considerations should guide {topic} development.",
                "Future trends in {topic} include AI and immersive technologies.",
                "Global adoption of {topic} has accelerated recently.",
                "Evidence-based research supports {topic} effectiveness."
            ],
            fact_templates=[
                "The global market for {topic} is valued at {number} billion.",
                "The number of {topic} users has grown by {percentage}%.",
                "Leading companies in {topic} include {company1} and {company2}.",
                "Key technologies in {topic} include {tech1} and {tech2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main applications of {topic}?",
                "How does {topic} transform education?",
                "What are the current trends in {topic}?",
                "What are the equity considerations for {topic}?",
                "What are the future directions of {topic}?"
            ]
        )
        
        # CleanTech templates
        templates['cleantech'] = DocumentTemplate(
            category='cleantech',
            topics=['solar energy', 'wind energy', 'energy storage', 'carbon capture', 'green hydrogen', 'circular economy', 'sustainable materials'],
            sentence_templates=[
                "{topic} addresses environmental challenges through innovation.",
                "Technology advances enhance {topic} efficiency and affordability.",
                "Policy support drives {topic} adoption.",
                "Investment in {topic} has increased significantly.",
                "Scalability is a key challenge for {topic} deployment.",
                "Integration with existing systems affects {topic} implementation.",
                "Global collaboration accelerates {topic} development.",
                "Future trends in {topic} include advanced materials and AI integration.",
                "Economic benefits accompany environmental benefits of {topic}.",
                "Public awareness supports {topic} adoption."
            ],
            fact_templates=[
                "Global investment in {topic} reached {number} billion last year.",
                "The capacity of {topic} has grown by {percentage}% annually.",
                "Leading countries in {topic} include {country1} and {country2}.",
                "Key applications of {topic} include {app1} and {app2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main applications of {topic}?",
                "How does {topic} address environmental challenges?",
                "What are the current trends in {topic}?",
                "What are the scalability challenges for {topic}?",
                "What are the future directions of {topic}?"
            ]
        )
        
        # Social Innovation templates
        templates['social_innovation'] = DocumentTemplate(
            category='social_innovation',
            topics=['social entrepreneurship', 'impact investing', 'community development', 'social finance', 'inclusive business', 'civic technology', 'social impact measurement'],
            sentence_templates=[
                "{topic} addresses social and environmental challenges through innovative solutions.",
                "Business models in {topic} balance profit and purpose.",
                "Impact measurement is essential for {topic} evaluation.",
                "Collaboration enhances {topic} effectiveness.",
                "Policy support enables {topic} scaling.",
                "Technology amplifies {topic} reach and impact.",
                "Community engagement is crucial for {topic} success.",
                "Future trends in {topic} include AI and blockchain applications.",
                "Investment in {topic} has grown significantly.",
                "Global networks support {topic} development."
            ],
            fact_templates=[
                "Global investment in {topic} reached {number} billion last year.",
                "The number of {topic} initiatives has grown by {percentage}%.",
                "Leading organizations in {topic} include {org1} and {org2}.",
                "Key areas of {topic} include {area1} and {area2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main approaches in {topic}?",
                "How does {topic} create social impact?",
                "What are the current trends in {topic}?",
                "How is {topic} measured and evaluated?",
                "What are the future directions of {topic}?"
            ]
        )
        
        # Urban Innovation templates
        templates['urban_innovation'] = DocumentTemplate(
            category='urban_innovation',
            topics=['smart mobility', 'urban farming', 'circular cities', 'resilient cities', 'participatory planning', 'urban analytics', 'community platforms'],
            sentence_templates=[
                "{topic} addresses urban challenges through innovative solutions.",
                "Technology enables new approaches to {topic}.",
                "Citizen participation enhances {topic} effectiveness.",
                "Sustainability is a key goal of {topic} initiatives.",
                "Data analytics supports {topic} decision-making.",
                "Pilot projects test {topic} innovations at scale.",
                "Policy frameworks support {topic} implementation.",
                "Future trends in {topic} include AI and IoT integration.",
                "Global cities are adopting {topic} solutions.",
                "Equity considerations should guide {topic} development."
            ],
            fact_templates=[
                "Global investment in {topic} reached {number} billion last year.",
                "The number of {topic} projects has grown by {percentage}%.",
                "Leading cities in {topic} include {city1} and {city2}.",
                "Key technologies in {topic} include {tech1} and {tech2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main approaches in {topic}?",
                "How does {topic} address urban challenges?",
                "What are the current trends in {topic}?",
                "How does technology enable {topic}?",
                "What are the future directions of {topic}?"
            ]
        )
        
        # FoodTech templates
        templates['foodtech'] = DocumentTemplate(
            category='foodtech',
            topics=['alternative proteins', 'food delivery', 'food waste reduction', 'precision agriculture', 'food safety technology', 'personalized nutrition', 'vertical farming'],
            sentence_templates=[
                "{topic} transforms how food is produced, distributed, and consumed.",
                "Technology enables innovative {topic} solutions.",
                "Sustainability is a key driver of {topic} innovation.",
                "Consumer demand shapes {topic} development.",
                "Investment in {topic} has increased significantly.",
                "Regulatory frameworks affect {topic} adoption.",
                "Integration with traditional food systems is important for {topic}.",
                "Future trends in {topic} include AI and biotechnology applications.",
                "Global adoption of {topic} is growing steadily.",
                "Health and environmental benefits drive {topic} acceptance."
            ],
            fact_templates=[
                "The global market for {topic} is valued at {number} billion.",
                "Investment in {topic} has grown by {percentage}% annually.",
                "Leading companies in {topic} include {company1} and {company2}.",
                "Key technologies in {topic} include {tech1} and {tech2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main applications of {topic}?",
                "How does {topic} transform food systems?",
                "What are the current trends in {topic}?",
                "What are the sustainability benefits of {topic}?",
                "What are the future directions of {topic}?"
            ]
        )
        
        # Water Technology templates
        templates['water_technology'] = DocumentTemplate(
            category='water_technology',
            topics=['water purification', 'desalination', 'water recycling', 'smart water management', 'leak detection', 'water quality monitoring', 'irrigation technology'],
            sentence_templates=[
                "{topic} addresses global water challenges through innovation.",
                "Technology enables efficient {topic} solutions.",
                "Scarcity drives investment in {topic} development.",
                "Sustainability is central to {topic} approaches.",
                "Integration with existing infrastructure affects {topic} implementation.",
                "Policy support enables {topic} adoption.",
                "Future trends in {topic} include AI and IoT integration.",
                "Global water challenges necessitate {topic} innovation.",
                "Cost considerations affect {topic} deployment.",
                "Public awareness supports {topic} acceptance."
            ],
            fact_templates=[
                "Global investment in {topic} reached {number} billion last year.",
                "The number of {topic} installations has grown by {percentage}%.",
                "Leading companies in {topic} include {company1} and {company2}.",
                "Key applications of {topic} include {app1} and {app2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main applications of {topic}?",
                "How does {topic} address water challenges?",
                "What are the current trends in {topic}?",
                "What are the sustainability benefits of {topic}?",
                "What are the future directions of {topic}?"
            ]
        )
        
        # Waste Technology templates
        templates['waste_technology'] = DocumentTemplate(
            category='waste_technology',
            topics=['recycling technology', 'waste-to-energy', 'composting systems', 'plastic recycling', 'e-waste recycling', 'circular economy solutions', 'zero waste initiatives'],
            sentence_templates=[
                "{topic} transforms waste management through innovation.",
                "Technology enables efficient {topic} processes.",
                "Circular economy principles guide {topic} development.",
                "Policy support drives {topic} adoption.",
                "Economic value can be recovered through {topic}.",
                "Environmental benefits motivate {topic} investment.",
                "Integration with existing systems affects {topic} implementation.",
                "Future trends in {topic} include AI and advanced materials.",
                "Global waste challenges necessitate {topic} innovation.",
                "Public participation supports {topic} success."
            ],
            fact_templates=[
                "The global market for {topic} is valued at {number} billion.",
                "The amount of waste processed by {topic} has grown by {percentage}%.",
                "Leading companies in {topic} include {company1} and {company2}.",
                "Key technologies in {topic} include {tech1} and {tech2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main applications of {topic}?",
                "How does {topic} transform waste management?",
                "What are the current trends in {topic}?",
                "What are the circular economy benefits of {topic}?",
                "What are the future directions of {topic}?"
            ]
        )
        
        # Climate Tech templates
        templates['climate_tech'] = DocumentTemplate(
            category='climate_tech',
            topics=['carbon capture', 'climate adaptation', 'climate monitoring', 'green finance', 'climate risk assessment', 'nature-based solutions', 'climate resilience'],
            sentence_templates=[
                "{topic} addresses climate change through technological innovation.",
                "Urgency drives investment in {topic} development.",
                "Policy support enables {topic} scaling.",
                "Technology advances enhance {topic} effectiveness.",
                "Global collaboration accelerates {topic} progress.",
                "Economic considerations affect {topic} adoption.",
                "Environmental impact is central to {topic} evaluation.",
                "Future trends in {topic} include AI and advanced materials.",
                "Climate challenges necessitate {topic} innovation.",
                "Public awareness supports {topic} acceptance."
            ],
            fact_templates=[
                "Global investment in {topic} reached {number} billion last year.",
                "The capacity of {topic} solutions has grown by {percentage}%.",
                "Leading countries in {topic} include {country1} and {country2}.",
                "Key applications of {topic} include {app1} and {app2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main applications of {topic}?",
                "How does {topic} address climate change?",
                "What are the current trends in {topic}?",
                "What are the policy considerations for {topic}?",
                "What are the future directions of {topic}?"
            ]
        )
        
        # Ocean Tech templates
        templates['ocean_tech'] = DocumentTemplate(
            category='ocean_tech',
            topics=['ocean monitoring', 'sustainable fishing', 'ocean energy', 'marine biotechnology', 'ocean cleanup', 'aquaculture technology', 'deep sea exploration'],
            sentence_templates=[
                "{topic} addresses ocean challenges through innovation.",
                "Technology enables new approaches to {topic}.",
                "Sustainability is central to {topic} development.",
                "Ocean health depends on {topic} solutions.",
                "Investment in {topic} has increased significantly.",
                "Regulatory frameworks govern {topic} activities.",
                "Future trends in {topic} include AI and autonomous systems.",
                "Global ocean challenges necessitate {topic} innovation.",
                "Collaboration enhances {topic} effectiveness.",
                "Public awareness supports {topic} acceptance."
            ],
            fact_templates=[
                "Global investment in {topic} reached {number} billion last year.",
                "The number of {topic} projects has grown by {percentage}%.",
                "Leading countries in {topic} include {country1} and {country2}.",
                "Key applications of {topic} include {app1} and {app2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main applications of {topic}?",
                "How does {topic} address ocean challenges?",
                "What are the current trends in {topic}?",
                "How does technology enable {topic}?",
                "What are the future directions of {topic}?"
            ]
        )
        
        # Forestry Tech templates
        templates['forestry_tech'] = DocumentTemplate(
            category='forestry_tech',
            topics=['sustainable forestry', 'forest monitoring', 'precision forestry', 'forest restoration', 'carbon sequestration', 'wood technology', 'forest fire management'],
            sentence_templates=[
                "{topic} enhances forest management through innovation.",
                "Technology enables efficient {topic} practices.",
                "Sustainability is essential for {topic} development.",
                "Climate change affects {topic} challenges and solutions.",
                "Investment in {topic} has increased significantly.",
                "Policy frameworks guide {topic} implementation.",
                "Future trends in {topic} include AI and remote sensing.",
                "Global forest challenges necessitate {topic} innovation.",
                "Economic and environmental benefits drive {topic} adoption.",
                "Community engagement supports {topic} success."
            ],
            fact_templates=[
                "Global investment in {topic} reached {number} billion last year.",
                "The area managed with {topic} has grown by {percentage}%.",
                "Leading countries in {topic} include {country1} and {country2}.",
                "Key technologies in {topic} include {tech1} and {tech2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main applications of {topic}?",
                "How does {topic} enhance forest management?",
                "What are the current trends in {topic}?",
                "How does technology enable {topic}?",
                "What are the future directions of {topic}?"
            ]
        )
        
        # Mining Tech templates
        templates['mining_tech'] = DocumentTemplate(
            category='mining_tech',
            topics=['sustainable mining', 'automated mining', 'mineral processing', 'mine safety', 'exploration technology', 'reclamation technology', 'deep sea mining'],
            sentence_templates=[
                "{topic} transforms mining operations through innovation.",
                "Technology enhances safety and efficiency in {topic}.",
                "Sustainability is increasingly important in {topic}.",
                "Automation is changing {topic} workforce requirements.",
                "Environmental considerations affect {topic} practices.",
                "Investment in {topic} has increased significantly.",
                "Regulatory frameworks govern {topic} operations.",
                "Future trends in {topic} include AI and robotics.",
                "Global demand drives {topic} development.",
                "Community engagement is important for {topic} acceptance."
            ],
            fact_templates=[
                "Global investment in {topic} reached {number} billion last year.",
                "The adoption of {topic} technologies has grown by {percentage}%.",
                "Leading companies in {topic} include {company1} and {company2}.",
                "Key technologies in {topic} include {tech1} and {tech2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main applications of {topic}?",
                "How does {topic} transform mining operations?",
                "What are the current trends in {topic}?",
                "What are the sustainability considerations for {topic}?",
                "What are the future directions of {topic}?"
            ]
        )
        
        # Construction Tech templates
        templates['construction_tech'] = DocumentTemplate(
            category='construction_tech',
            topics=['modular construction', '3D printing in construction', 'smart buildings', 'construction robotics', 'BIM technology', 'sustainable construction', 'construction safety'],
            sentence_templates=[
                "{topic} transforms construction through innovation.",
                "Technology enhances efficiency and safety in {topic}.",
                "Sustainability is increasingly important in {topic}.",
                "Prefabrication is changing {topic} processes.",
                "Digital tools enable better {topic} planning.",
                "Investment in {topic} has increased significantly.",
                "Regulatory frameworks affect {topic} adoption.",
                "Future trends in {topic} include AI and robotics.",
                "Global construction challenges drive {topic} innovation.",
                "Workforce considerations affect {topic} implementation."
            ],
            fact_templates=[
                "The global market for {topic} is valued at {number} billion.",
                "The adoption of {topic} technologies has grown by {percentage}%.",
                "Leading companies in {topic} include {company1} and {company2}.",
                "Key technologies in {topic} include {tech1} and {tech2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main applications of {topic}?",
                "How does {topic} transform construction?",
                "What are the current trends in {topic}?",
                "What are the sustainability benefits of {topic}?",
                "What are the future directions of {topic}?"
            ]
        )
        
        return templates
            category='environment',
            topics=['renewable energy', 'climate change mitigation', 'sustainable agriculture', 'waste management', 'biodiversity conservation', 'water conservation', 'green building'],
            sentence_templates=[
                "Sustainable practices in {topic} are crucial for environmental protection.",
                "Global initiatives in {topic} have gained momentum in recent years.",
                "The impact of {topic} on ecosystems is significant.",
                "Innovations in {topic} are making environmental solutions more accessible.",
                "Public awareness of {topic} has increased substantially.",
                "Economic benefits of {topic} include job creation and cost savings.",
                "Policy frameworks for {topic} are evolving globally.",
                "Technology plays a key role in advancing {topic} solutions.",
                "Community engagement is essential for successful {topic} implementation.",
                "The future of {topic} depends on continued innovation and investment."
            ],
            fact_templates=[
                "Global investment in {topic} reached {number} billion last year.",
                "The adoption of {topic} has reduced emissions by {percentage}%.",
                "Leading countries in {topic} include {country1} and {country2}.",
                "Key benefits of {topic} include {benefit1} and {benefit2}.",
                "Major challenges in {topic} include {challenge1} and {challenge2}."
            ],
            question_templates=[
                "What are the main benefits of {topic}?",
                "How can communities implement {topic}?",
                "What are the latest innovations in {topic}?",
                "What policies support {topic}?",
                "How does {topic} impact the environment?"
            ]
        )
        
        return templates
    
    def generate_document(self, category: str, topic: str, length: int = 5) -> str:
        """
        Generate a single document on a specific topic.
        
        Args:
            category: Document category
            topic: Specific topic within category
            length: Number of paragraphs
            
        Returns:
            Generated document text
        """
        if category not in self.templates:
            category = random.choice(list(self.templates.keys()))
        
        template = self.templates[category]
        paragraphs = []
        
        for i in range(length):
            # Mix different template types
            template_type = random.choice(['sentence', 'fact', 'question'])
            
            if template_type == 'sentence':
                sentence = random.choice(template.sentence_templates)
                paragraphs.append(sentence.format(topic=topic))
            elif template_type == 'fact':
                fact = random.choice(template.fact_templates)
                paragraphs.append(fact.format(
                    topic=topic,
                    decade=random.choice(['1990s', '2000s', '2010s', '2020s']),
                    year=random.randint(2010, 2030),
                    number=random.randint(10, 1000),
                    company1=random.choice(['Google', 'Microsoft', 'Amazon', 'Apple', 'Meta']),
                    company2=random.choice(['IBM', 'Oracle', 'Salesforce', 'Adobe', 'Intel']),
                    company3=random.choice(['Tesla', 'Netflix', 'Spotify', 'Uber', 'Airbnb']),
                    application1=random.choice(['automation', 'data analysis', 'communication', 'security']),
                    application2=random.choice(['optimization', 'prediction', 'monitoring', 'control']),
                    application3=random.choice(['integration', 'scalability', 'efficiency', 'innovation']),
                    challenge1=random.choice(['security', 'cost', 'complexity', 'adoption']),
                    challenge2=random.choice(['scalability', 'maintenance', 'integration', 'regulation']),
                    institution1=random.choice(['MIT', 'Stanford', 'Harvard', 'Oxford', 'Cambridge']),
                    institution2=random.choice(['Caltech', 'Princeton', 'Yale', 'Imperial College', 'ETH Zurich']),
                    tool1=random.choice(['AI', 'machine learning', 'data analytics', 'simulation']),
                    tool2=random.choice(['cloud computing', 'big data', 'IoT', 'blockchain']),
                    focus_area=random.choice(['practical applications', 'theoretical research', 'experimental studies']),
                    metric1=random.choice(['ROI', 'efficiency', 'customer satisfaction', 'market share']),
                    metric2=random.choice(['growth rate', 'profit margin', 'employee engagement', 'innovation index']),
                    percentage=random.randint(10, 90),
                    org1=random.choice(['WHO', 'CDC', 'NIH', 'FDA', 'UN']),
                    org2=random.choice(['Red Cross', 'Doctors Without Borders', 'UNICEF', 'World Bank']),
                    recommendation1=random.choice(['regular exercise', 'balanced diet', 'stress management']),
                    recommendation2=random.choice(['adequate sleep', 'social connection', 'preventive care']),
                    disease=random.choice(['heart disease', 'diabetes', 'cancer', 'stroke']),
                    skill1=random.choice(['critical thinking', 'problem-solving', 'communication']),
                    skill2=random.choice(['collaboration', 'creativity', 'adaptability']),
                    country1=random.choice(['Germany', 'China', 'Japan', 'Sweden', 'Denmark']),
                    country2=random.choice(['Norway', 'Finland', 'Netherlands', 'Switzerland', 'Canada']),
                    benefit1=random.choice(['cost savings', 'environmental protection', 'health benefits']),
                    benefit2=random.choice(['job creation', 'energy independence', 'resource efficiency'])
                ))
            elif template_type == 'question':
                question = random.choice(template.question_templates)
                paragraphs.append(question.format(topic=topic))
        
        return '\n\n'.join(paragraphs)
    
    def generate_batch(self, num_documents: int, categories: List[str] = None) -> List[Tuple[str, str]]:
        """
        Generate a batch of documents.
        
        Args:
            num_documents: Number of documents to generate
            categories: List of categories to use (random if None)
            
        Returns:
            List of (doc_id, content) tuples
        """
        if categories is None:
            categories = list(self.templates.keys())
        
        documents = []
        
        for i in range(num_documents):
            category = random.choice(categories)
            topic = random.choice(self.templates[category].topics)
            length = random.randint(3, 8)
            
            content = self.generate_document(category, topic, length)
            doc_id = f"{category}_{topic}_{i}"
            
            documents.append((doc_id, content))
        
        return documents
    
    def save_documents(self, documents: List[Tuple[str, str]], batch_size: int = 1000):
        """
        Save generated documents to files.
        
        Args:
            documents: List of (doc_id, content) tuples
            batch_size: Number of documents per file
        """
        batch_num = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            filename = f"batch_{batch_num}.txt"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                for doc_id, content in batch:
                    f.write(f"[{doc_id}]\n{content}\n\n---\n\n")
            
            batch_num += 1
            print(f"Saved batch {batch_num} with {len(batch)} documents")
    
    def generate_large_scale(self, target_count: int, batch_size: int = 10000) -> int:
        """
        Generate large-scale document collection.
        
        Args:
            target_count: Target number of documents
            batch_size: Number of documents per generation batch
            
        Returns:
            Actual number of documents generated
        """
        total_generated = 0
        categories = list(self.templates.keys())
        
        while total_generated < target_count:
            batch_count = min(batch_size, target_count - total_generated)
            documents = self.generate_batch(batch_count, categories)
            
            self.save_documents(documents, batch_size=1000)
            total_generated += len(documents)
            
            print(f"Progress: {total_generated}/{target_count} documents generated")
        
        return total_generated


def generate_diverse_documents(target_count: int = 10000):
    """
    Generate diverse documents for knowledge base expansion.
    
    Args:
        target_count: Number of documents to generate
    """
    generator = DocumentGenerator()
    generated = generator.generate_large_scale(target_count)
    print(f"\nGeneration complete: {generated} documents created")
    return generated


if __name__ == "__main__":
    # Generate 10,000 documents as a starting point
    generate_diverse_documents(10000)
