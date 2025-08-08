import requests
from typing import Dict, List, Optional
from loguru import logger
import json
import re
from datetime import datetime
from django.conf import settings
from langchain_openai import ChatOpenAI


class KNUSTTools:
    """Tools for providing KNUST information and answering questions about KNUST"""
    
    def __init__(self):
        self.knust_website = "https://www.knust.edu.gh/"
        self.knust_info = self._get_knust_basic_info()
        self.llm = ChatOpenAI(
            model=getattr(settings, 'OPENAI_MODEL', 'gpt-3.5-turbo'),
            openai_api_key=getattr(settings, 'OPENAI_API_KEY', ''),
            temperature=0.3
        )
    
    def _get_knust_basic_info(self) -> Dict:
        """Get basic KNUST information"""
        return {
            "name": "Kwame Nkrumah University of Science and Technology",
            "location": "Kumasi, Ghana",
            "established": "1952",
            "type": "Public University",
            "motto": "Nyansapo Wosane No Badwemma",
            "website": "https://www.knust.edu.gh/",
            "contact": {
                "phone": "+233 5000 99299",
                "email": "uro@knust.edu.gh",
                "gps_address": "AK-385-1973"
            },
            "colleges": [
                "College of Science",
                "College of Engineering", 
                "College of Art & Built Environment",
                "College of Agric & Natural Resources",
                "College of Health Sciences",
                "College of Humanities & Social Sciences"
            ],
            "departments": [
                "School of Graduate Studies",
                "Institute of Distance Learning",
                "University Library",
                "Office of Grants & Research",
                "Quality Assurance & Planning",
                "Human Resource Development"
            ],
            "services": [
                "Students Portal",
                "Staff Portal", 
                "Admissions Portal",
                "E-Learning Centre",
                "Library Services",
                "Career Services Centre"
            ],
            "programs": {
                "undergraduate": "Bachelor's degrees across all 6 colleges",
                "graduate": "Master's degrees (MSc, MA, MPhil) and Doctoral programs (PhD)",
                "distance_learning": "Institute of Distance Learning (IDL) with flexible learning options",
                "research": "PhD programs across all colleges with research centers and institutes"
            },
            "admissions": {
                "undergraduate": "Applications typically open from August to September",
                "graduate": "School of Graduate Studies handles postgraduate applications",
                "international": "International students have separate application processes",
                "distance_learning": "Distance Learning programs available for working professionals"
            },
            "academic_calendar": "2024/2025 Academic Calendar available with semester-based academic year",
            "research_centers": [
                "Office of Grants & Research (OGR)",
                "Research funding opportunities",
                "Research centers and institutes",
                "Publication support"
            ],
            "student_life": {
                "accommodation": "Campus accommodation options available",
                "sports": "Sports complexes and gymnasiums",
                "facilities": "Modern lecture halls and laboratories",
                "organizations": "Student organizations and clubs"
            },
            "financial_aid": {
                "scholarships": "Students' Financial Services Office (SFSO) offers various scholarships",
                "bursaries": "Bursaries and financial aid opportunities",
                "work_study": "Work-study opportunities available"
            }
        }
    
    def get_knust_info(self, query: str) -> str:
        """Get intelligent KNUST information based on question intent"""
        try:
            # Use AI to understand the question and provide relevant information
            prompt = self._create_intelligent_prompt(query)
            response = self.llm.invoke(prompt)
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error getting KNUST info: {str(e)}")
            # Fallback to general info if AI fails
            return self._get_general_info()
    
    def _create_intelligent_prompt(self, query: str) -> str:
        """Create an intelligent prompt for understanding KNUST questions"""
        knust_data = json.dumps(self.knust_info, indent=2)
        
        return f"""You are a helpful assistant for Kwame Nkrumah University of Science and Technology (KNUST). 
You have access to comprehensive information about KNUST and should provide accurate, helpful responses.

KNUST Information Database:
{knust_data}

User Question: "{query}"

Instructions:
1. Analyze the user's question carefully to understand their intent
2. Provide a relevant, accurate answer based on the KNUST information available
3. If the question is not related to KNUST, politely redirect them
4. If you don't have enough information to answer the question, say so
5. Format your response clearly and professionally
6. Use emojis where appropriate to make the response engaging
7. If the question is about admissions, programs, contact info, etc., provide specific details
8. If the question is too vague, ask for clarification

Please provide a helpful response to the user's question about KNUST."""
    
    def _get_contact_info(self) -> str:
        """Get KNUST contact information"""
        return f"""
**KNUST Contact Information:**

📍 **Location:** Kumasi, Ghana
📞 **Phone:** {self.knust_info['contact']['phone']}
📧 **Email:** {self.knust_info['contact']['email']}
🌍 **Website:** {self.knust_info['website']}
📍 **GPS Address:** {self.knust_info['contact']['gps_address']}

For specific inquiries, you can contact the University Relations Office (URO) at the provided email address.
        """.strip()
    
    def _get_colleges_info(self) -> str:
        """Get information about KNUST colleges"""
        colleges = "\n".join([f"• {college}" for college in self.knust_info['colleges']])
        return f"""
**KNUST Colleges:**

KNUST has 6 main colleges offering various academic programs:

{colleges}

Each college contains multiple departments and schools offering undergraduate and postgraduate programs in their respective fields of study.
        """.strip()
    
    def _get_admission_info(self) -> str:
        """Get admission information"""
        return f"""
**KNUST Admission Information:**

**Undergraduate Admissions:**
• Applications are typically open from August to September
• Cut-off points vary by program and year
• International students have separate application processes
• Distance Learning programs are also available

**Graduate Admissions:**
• School of Graduate Studies handles postgraduate applications
• Various Master's and PhD programs available
• Research-based and taught programs offered

**How to Apply:**
1. Visit the official website: {self.knust_info['website']}
2. Navigate to the Admissions section
3. Check current admission requirements and cut-off points
4. Submit application through the online portal
5. Pay required application fees

**Contact Admissions:**
📧 Email: uro@knust.edu.gh
📞 Phone: {self.knust_info['contact']['phone']}
        """.strip()
    
    def _get_programs_info(self) -> str:
        """Get information about academic programs"""
        return f"""
**KNUST Academic Programs:**

**Undergraduate Programs:**
• Bachelor's degrees across all 6 colleges
• Programs in Science, Engineering, Arts, Health Sciences, Agriculture, and Humanities
• Duration: 3-4 years depending on program

**Graduate Programs:**
• Master's degrees (MSc, MA, MPhil)
• Doctoral programs (PhD)
• Professional programs and certificates

**Distance Learning:**
• Institute of Distance Learning (IDL)
• Flexible learning options for working professionals
• Various undergraduate and graduate programs

**Research Programs:**
• PhD programs across all colleges
• Research centers and institutes
• Collaboration with international institutions

For specific program details, visit the official website or contact the respective college/department.
        """.strip()
    
    def _get_academic_services_info(self) -> str:
        """Get information about academic services"""
        return f"""
**KNUST Academic Services:**

**University Library:**
• Main library with extensive collections
• Digital resources and databases
• Research support services
• Study spaces and computer labs

**Research Support:**
• Office of Grants & Research (OGR)
• Research funding opportunities
• Research centers and institutes
• Publication support

**Academic Calendar:**
• 2024/2025 Academic Calendar available
• Semester-based academic year
• Regular updates on academic schedules

**Quality Assurance:**
• Quality Assurance & Planning Office
• Academic standards monitoring
• Program accreditation
• Continuous improvement initiatives

**E-Learning:**
• E-Learning Centre
• Online course platforms
• Digital learning resources
• Technology-enhanced teaching
        """.strip()
    
    def _get_student_services_info(self) -> str:
        """Get information about student services"""
        return f"""
**KNUST Student Services:**

**Student Portal:**
• Online registration and course selection
• Academic records and transcripts
• Fee payment and financial aid
• Exam schedules and results

**Student Life:**
• Directorate of Student Affairs
• Student organizations and clubs
• Campus accommodation options
• Sports and recreational facilities

**Financial Services:**
• Students' Financial Services Office (SFSO)
• Scholarships and bursaries
• Student loan programs
• Work-study opportunities

**Support Services:**
• Career Services Centre
• Counseling and guidance
• Health services
• International student support

**Campus Facilities:**
• Modern lecture halls and laboratories
• Sports complexes and gymnasiums
• Student centers and cafeterias
• Transportation services
        """.strip()
    
    def _get_staff_info(self) -> str:
        """Get information about staff services"""
        return f"""
**KNUST Staff Services:**

**Staff Portal:**
• Online staff directory
• Administrative tools and resources
• HR services and benefits
• Professional development opportunities

**Employment:**
• Job application portal
• Current vacancies and openings
• Academic and non-academic positions
• International staff recruitment

**Staff Development:**
• Human Resource Development Office
• Training and workshops
• Professional certification programs
• Research and publication support

**Staff Benefits:**
• Competitive salary packages
• Health insurance coverage
• Housing allowances
• Professional development funding

**Contact HR:**
📧 Email: hr@knust.edu.gh
📞 Phone: {self.knust_info['contact']['phone']}
        """.strip()
    
    def _get_general_info(self) -> str:
        """Get general KNUST information"""
        return f"""
**About KNUST - Kwame Nkrumah University of Science and Technology**

**Basic Information:**
• **Name:** {self.knust_info['name']}
• **Location:** {self.knust_info['location']}
• **Established:** {self.knust_info['established']}
• **Type:** {self.knust_info['type']}
• **Motto:** "{self.knust_info['motto']}"

**Overview:**
KNUST is one of Ghana's premier universities, specializing in science and technology education. The university offers a wide range of undergraduate and postgraduate programs across six colleges, serving both local and international students.

**Key Features:**
• 6 main colleges with diverse academic programs
• Strong focus on science, technology, and innovation
• Research-intensive institution
• International partnerships and collaborations
• Modern campus facilities and infrastructure

**Contact Information:**
📞 Phone: {self.knust_info['contact']['phone']}
📧 Email: {self.knust_info['contact']['email']}
🌍 Website: {self.knust_info['website']}
📍 GPS: {self.knust_info['contact']['gps_address']}

For more specific information, please ask about admissions, programs, colleges, or any particular aspect of the university.
        """.strip()
    
    def search_knust_website(self, query: str) -> Optional[str]:
        """Search KNUST website for specific information (placeholder)"""
        try:
            # This would ideally use web scraping or API calls to search the actual website
            # For now, we'll return None and rely on our predefined information
            logger.info(f"Search query for KNUST website: {query}")
            return None
        except Exception as e:
            logger.error(f"Error searching KNUST website: {str(e)}")
            return None
    
    def get_current_news(self) -> str:
        """Get current KNUST news and announcements"""
        return """
**Recent KNUST News and Updates:**

• **2025/2026 Admissions:** Application e-vouchers for Postgraduate/Undergraduate/Top-Up programmes are now on sale
• **Professorial Inaugural Lecture:** Professor Emmanuel Adinyira will deliver a lecture on August 21, 2025
• **Research Collaboration:** KNUST collaborates with ARIPO to promote innovation through intellectual property
• **Student Support:** Various scholarship programs and financial aid opportunities available
• **Campus Development:** Ongoing infrastructure improvements and facility upgrades

For the latest news and announcements, visit the official KNUST website or contact the University Relations Office.
        """.strip()
    
    def is_knust_related_question(self, question: str) -> bool:
        """Intelligently check if a question is related to KNUST"""
        try:
            # Use AI to determine if the question is KNUST-related
            prompt = f"""Analyze the following question and determine if it's related to Kwame Nkrumah University of Science and Technology (KNUST).

Question: "{question}"

Instructions:
1. Consider if the question is asking about KNUST specifically
2. Consider if it's asking about university life, admissions, academic programs, etc. that could be KNUST-related
3. Consider if it's asking about Ghanaian universities or education in Ghana
4. If the question is too vague or could be about any university, it's NOT KNUST-specific
5. If the question is clearly about something else (like weather, sports, etc.), it's NOT KNUST-related

Respond with ONLY "YES" if the question is KNUST-related, or "NO" if it's not."""

            response = self.llm.invoke(prompt)
            result = response.content.strip().upper()
            
            return result == "YES"
            
        except Exception as e:
            logger.error(f"Error checking KNUST relation: {str(e)}")
            # Fallback to keyword-based detection
            knust_keywords = [
                'knust', 'kwame nkrumah', 'kumasi', 'ghana university',
                'admission', 'apply', 'college', 'faculty', 'program',
                'student', 'staff', 'library', 'research', 'campus',
                'lecture', 'exam', 'course', 'degree', 'scholarship'
            ]
            
            question_lower = question.lower()
            return any(keyword in question_lower for keyword in knust_keywords)


# Create a global instance
knust_tools = KNUSTTools() 