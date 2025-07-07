#!/usr/bin/env python3
"""
refined_diarized_summarizer.py

REFINED DIARIZED INDIAN LEGAL SUMMARIZATION SYSTEM
Addresses critical issues:
- Fixed text truncation problems
- Better speaker name extraction
- Improved content length handling
- Enhanced legal section extraction
- Complete evidence details capture
- Proper speaker label parsing

Usage:
  python refined_diarized_summarizer.py --input diarized_proceedings.txt --output summary.txt
"""

import argparse, logging, sys, re
from typing import Dict, List, Tuple

# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser("Refined Diarized Indian Legal Summarizer")
parser.add_argument("-i","--input", required=True, help="Input diarized transcript file")
parser.add_argument("-o","--output", required=True, help="Output summary file")
parser.add_argument("--max-narrative", type=int, default=600, help="Max narrative summary length")
parser.add_argument("--max-content", type=int, default=500, help="Max content length per section")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Enhanced Diarized Content Parser
# -----------------------------------------------------------------------------
class EnhancedDiarizedParser:
    """Enhanced parser with better speaker recognition and content handling"""
    
    def __init__(self):
        self.max_content_length = args.max_content
    
    def parse_enhanced_diarized_content(self, text: str) -> Dict:
        """Enhanced parsing with better speaker recognition"""
        
        parsed_content = {
            'judge_statements': [],
            'prosecutor_statements': [],
            'defense_statements': [],
            'witness_statements': [],
            'clerk_statements': [],
            'speaker_mapping': {},
            'all_speakers': [],
            'personnel_info': self._extract_personnel_info(text)
        }
        
        # Split text into speaker segments with better handling
        speaker_segments = self._enhanced_split_by_speakers(text)
        
        for speaker, content in speaker_segments:
            speaker_type = self._enhanced_identify_speaker_type(speaker)
            cleaned_content = self._enhanced_clean_speaker_content(content)
            
            # Store content without truncation initially
            if speaker_type == 'judge':
                parsed_content['judge_statements'].append(cleaned_content)
            elif speaker_type == 'prosecutor':
                parsed_content['prosecutor_statements'].append(cleaned_content)
            elif speaker_type == 'defense':
                parsed_content['defense_statements'].append(cleaned_content)
            elif speaker_type == 'witness':
                parsed_content['witness_statements'].append(cleaned_content)
            elif speaker_type == 'clerk':
                parsed_content['clerk_statements'].append(cleaned_content)
            
            parsed_content['speaker_mapping'][speaker] = speaker_type
            if speaker not in parsed_content['all_speakers']:
                parsed_content['all_speakers'].append(speaker)
        
        return parsed_content
    
    def _extract_personnel_info(self, text: str) -> Dict:
        """Extract personnel information from 'Present:' section"""
        
        personnel = {
            'public_prosecutor': '',
            'defense_advocates': [],
            'judge': ''
        }
        
        # Extract from Present section
        present_match = re.search(r'Present:(.*?)(?=PROCEEDINGS:|$)', text, re.DOTALL)
        if present_match:
            present_text = present_match.group(1)
            
            # Public Prosecutor
            pp_patterns = [
                r'Shri ([A-Za-z\s]+), Public Prosecutor',
                r'Shri ([A-Za-z\s]+), Additional Public Prosecutor',
                r'Ms\. ([A-Za-z\s]+), Public Prosecutor'
            ]
            
            for pattern in pp_patterns:
                match = re.search(pattern, present_text)
                if match:
                    personnel['public_prosecutor'] = f"Shri {match.group(1).strip()}, Public Prosecutor"
                    break
            
            # Defense Advocates
            advocate_patterns = [
                r'Shri Advocate ([A-Za-z\s]+) for (?:Accused|Defendant|Petitioner|Respondent)',
                r'Ms\. Advocate ([A-Za-z\s]+) for (?:Accused|Defendant|Petitioner|Respondent)',
                r'Shri Advocate ([A-Za-z\s]+) for the (?:plaintiff|defendant)',
                r'Ms\. Advocate ([A-Za-z\s]+) for the (?:plaintiff|defendant)'
            ]
            
            for pattern in advocate_patterns:
                matches = re.findall(pattern, present_text)
                for match in matches:
                    advocate_name = f"Advocate {match.strip()}"
                    if advocate_name not in personnel['defense_advocates']:
                        personnel['defense_advocates'].append(advocate_name)
        
        # Extract Judge from Coram section
        judge_patterns = [
            r'Hon\'ble Shri Justice ([A-Za-z\s\.]+)',
            r'Hon\'ble Ms\. Justice ([A-Za-z\s\.]+)',
            r'Coram: Hon\'ble ([^\n]+)'
        ]
        
        for pattern in judge_patterns:
            match = re.search(pattern, text)
            if match:
                personnel['judge'] = match.group(1).strip()
                break
        
        return personnel
    
    def _enhanced_split_by_speakers(self, text: str) -> List[Tuple[str, str]]:
        """Enhanced speaker splitting with better pattern recognition"""
        
        # More comprehensive speaker patterns
        speaker_patterns = [
            r'\[([A-Z_][A-Z_\s]*)\]:\s*',  # [SPEAKER]:
            r'([A-Z_][A-Z_\s]*):(?=\s)',   # SPEAKER: (with space after)
        ]
        
        segments = []
        current_speaker = None
        current_content = []
        
        lines = text.split('\n')
        
        for line in lines:
            speaker_found = False
            
            for pattern in speaker_patterns:
                speaker_match = re.match(pattern, line)
                if speaker_match:
                    # Save previous speaker's content
                    if current_speaker and current_content:
                        content = '\n'.join(current_content)
                        if len(content.strip()) > 10:  # Only meaningful content
                            segments.append((current_speaker, content))
                    
                    # Start new speaker
                    current_speaker = speaker_match.group(1).strip()
                    remaining_line = line[speaker_match.end():].strip()
                    current_content = [remaining_line] if remaining_line else []
                    speaker_found = True
                    break
            
            if not speaker_found and current_speaker:
                current_content.append(line)
        
        # Add last speaker's content
        if current_speaker and current_content:
            content = '\n'.join(current_content)
            if len(content.strip()) > 10:
                segments.append((current_speaker, content))
        
        return segments
    
    def _enhanced_identify_speaker_type(self, speaker: str) -> str:
        """Enhanced speaker type identification"""
        
        speaker_lower = speaker.lower().replace('_', ' ')
        
        # Judge patterns
        if any(word in speaker_lower for word in ['judge', 'court', 'lordship']):
            return 'judge'
        
        # Prosecutor patterns
        elif any(word in speaker_lower for word in ['prosecutor', 'app', 'public']):
            return 'prosecutor'
        
        # Defense/Advocate patterns
        elif any(word in speaker_lower for word in ['advocate', 'defense', 'counsel']):
            return 'defense'
        
        # Clerk patterns
        elif any(word in speaker_lower for word in ['clerk', 'registrar']):
            return 'clerk'
        
        # Witness patterns (specific names)
        elif any(word in speaker_lower for word in ['inspector', 'doctor', 'dr', 'constable', 'smt', 'shri']):
            return 'witness'
        
        # Default to witness for other speakers
        else:
            return 'witness'
    
    def _enhanced_clean_speaker_content(self, content: str) -> str:
        """Enhanced content cleaning without aggressive truncation"""
        
        # Remove speaker labels from content
        content = re.sub(r'\[[A-Z_\s]+\]:\s*', '', content)
        content = re.sub(r'^[A-Z_\s]+:\s*', '', content, flags=re.MULTILINE)
        
        # Clean up excessive whitespace but preserve structure
        content = re.sub(r'\n\s*\n', '\n', content)  # Remove empty lines
        content = re.sub(r'[ \t]+', ' ', content)     # Normalize spaces
        content = content.strip()
        
        return content

# -----------------------------------------------------------------------------
# Refined Information Extractor
# -----------------------------------------------------------------------------
class RefinedDiarizedExtractor:
    """Refined extractor with better information capture"""
    
    def __init__(self, case_type: str):
        self.case_type = case_type
        self.parser = EnhancedDiarizedParser()
    
    def extract_refined_case_info(self, text: str) -> Dict:
        """Extract comprehensive information with better accuracy"""
        
        # Parse diarized content with enhancements
        parsed_content = self.parser.parse_enhanced_diarized_content(text)
        
        info = {
            # Basic case information - enhanced extraction
            'case_name': self._extract_case_name_refined(text),
            'case_number': self._extract_case_number_refined(text),
            'case_type': self.case_type,
            'court_name': self._extract_court_name_refined(text),
            'judge_name': self._extract_judge_name_refined(text, parsed_content),
            'hearing_date': self._extract_hearing_date_refined(text),
            
            # Personnel - from parsed content and text
            'public_prosecutor': self._extract_prosecutor_refined(parsed_content),
            'defense_advocates': self._extract_defense_refined(parsed_content),
            'witnesses': self._extract_witnesses_refined(parsed_content),
            
            # Case parties - enhanced patterns
            'accused_persons': self._extract_accused_refined(text, parsed_content),
            'complainant': self._extract_complainant_refined(text, parsed_content),
            'victim': self._extract_victim_refined(text, parsed_content),
            
            # Legal specifics - comprehensive extraction
            'fir_details': self._extract_fir_refined(text, parsed_content),
            'police_station': self._extract_police_station_refined(text, parsed_content),
            'legal_sections': self._extract_legal_sections_refined(text, parsed_content),
            
            # Content from speakers - full content
            'judge_observations': self._extract_judge_content_refined(parsed_content),
            'prosecutor_arguments': self._extract_prosecutor_content_refined(parsed_content),
            'defense_arguments': self._extract_defense_content_refined(parsed_content),
            'witness_testimony': self._extract_witness_content_refined(parsed_content),
            'case_law_cited': self._extract_case_law_refined(text, parsed_content),
            
            # Proceedings - detailed extraction
            'evidence_presented': self._extract_evidence_refined(text, parsed_content),
            'key_facts': self._extract_key_facts_refined(text, parsed_content),
            'next_date': self._extract_next_date_refined(text),
            'judgment_status': self._extract_judgment_status_refined(text)
        }
        
        return info

    def _extract_case_name_refined(self, text: str) -> str:
        """Refined case name extraction with better patterns"""

        patterns = [
            # Look for case name after case number
            r'Sessions Case No\. \d+/\d+\s*\n\s*([^\n]+?)(?:\s*\n|\s*Coram)',
            r'Civil Suit No\. \d+/\d+\s*\n\s*([^\n]+?)(?:\s*\n|\s*Coram)',
            r'Matrimonial Case No\. \d+/\d+\s*\n\s*([^\n]+?)(?:\s*\n|\s*Coram)',
            r'Criminal Appeal No\. \d+/\d+\s*\n\s*([^\n]+?)(?:\s*\n|\s*Coram)',

            # Direct patterns
            r'State \(NCT of Delhi\) v\. ([^\n\r]+?)(?:\s*\n|\s*$)',
            r'State of ([A-Za-z\s]+) v\. ([^\n\r]+?)(?:\s*\n|\s*$)',
            r'(Smt\. [A-Za-z\s\.]+) v\. (Shri [A-Za-z\s\.]+)',
            r'(Shri [A-Za-z\s\.]+) v\. (M/s [A-Za-z\s\.&]+)',
            r'([A-Za-z\s\.]+) v\. ([A-Za-z\s\.&]+) & Anr',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
            if match:
                if len(match.groups()) == 2:  # Two parties
                    party1 = match.group(1).strip()
                    party2 = match.group(2).strip()
                    return f"{party1} v. {party2}"
                else:
                    case_name = match.group(1).strip()
                    # Clean up the case name
                    case_name = re.sub(r'\s+', ' ', case_name)
                    case_name = re.sub(r'\n.*', '', case_name)  # Remove anything after newline
                    return case_name

        return "Case name not identified"

    def _extract_case_number_refined(self, text: str) -> str:
        """Refined case number extraction"""

        patterns = [
            r'Sessions Case No\. (\d+/\d+)',
            r'Civil Suit No\. (\d+/\d+)',
            r'Matrimonial Case No\. (\d+/\d+)',
            r'Criminal Appeal No\. (\d+/\d+)',
            r'Writ Petition No\. (\d+/\d+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                case_type_full = pattern.split('\\')[0].replace('(', '').strip()
                return f"{case_type_full} {match.group(1)}"

        return "Case number not identified"

    def _extract_court_name_refined(self, text: str) -> str:
        """Refined court name extraction without truncation"""

        patterns = [
            r'IN THE COURT OF ([^\n]+?)(?=\n[A-Z]|\nCoram|\n\n)',
            r'IN THE ([A-Z\s]+COURT[^\n]*?)(?=\n[A-Z]|\nCoram|\n\n)',
            r'(FAMILY COURT[^\n]*?)(?=\n[A-Z]|\nCoram|\n\n)',
            r'(HIGH COURT OF [^\n]*?)(?=\n[A-Z]|\nCoram|\n\n)'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                court_name = match.group(1).strip()
                # Clean up multiple spaces
                court_name = re.sub(r'\s+', ' ', court_name)
                return court_name

        return "Court not identified"

    def _extract_judge_name_refined(self, text: str, parsed_content: Dict) -> str:
        """Refined judge name extraction"""

        # First try from personnel info
        personnel_judge = parsed_content.get('personnel_info', {}).get('judge', '')
        if personnel_judge:
            return personnel_judge

        # Fallback to pattern matching
        patterns = [
            r'Hon\'ble Shri Justice ([A-Za-z\s\.]+?)(?=\n|Additional|District|Sessions|Family)',
            r'Hon\'ble Ms\. Justice ([A-Za-z\s\.]+?)(?=\n|Additional|District|Sessions|Family)',
            r'Coram: Hon\'ble ([^\n]+?)(?=\n[A-Z]|\n\n)',
            r'Sessions Judge[:\s]*([A-Za-z\s\.]+?)(?=\n|Date)',
            r'Family Court Judge[:\s]*([A-Za-z\s\.]+?)(?=\n|Date)'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                judge_name = match.group(1).strip()
                # Clean up the name
                judge_name = re.sub(r'\s+', ' ', judge_name)
                judge_name = re.sub(r'\n.*', '', judge_name)  # Remove anything after newline
                return judge_name

        return "Judge not identified"

    def _extract_hearing_date_refined(self, text: str) -> str:
        """Refined hearing date extraction"""

        patterns = [
            r'Date:\s*([^\n]+?)(?=\n|Present)',
            r'Date of Hearing:\s*([^\n]+?)(?=\n)',
            r'(\d{1,2}[a-z]{2}\s+[A-Za-z]+,?\s+\d{4})'
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                date = match.group(1).strip()
                # Clean up date
                date = re.sub(r'\s+', ' ', date)
                return date

        return "Date not specified"

    def _extract_prosecutor_refined(self, parsed_content: Dict) -> str:
        """Refined prosecutor extraction"""

        # First try from personnel info
        personnel_pp = parsed_content.get('personnel_info', {}).get('public_prosecutor', '')
        if personnel_pp:
            return personnel_pp

        # Look for prosecutor in speaker mapping
        for speaker, speaker_type in parsed_content['speaker_mapping'].items():
            if speaker_type == 'prosecutor':
                # Clean up speaker name
                clean_name = speaker.replace('_', ' ').replace('PUBLIC PROSECUTOR', 'Public Prosecutor')
                return f"Shri {clean_name}"

        return "Public Prosecutor not identified"

    def _extract_defense_refined(self, parsed_content: Dict) -> List[str]:
        """Refined defense advocates extraction"""

        # First try from personnel info
        personnel_advocates = parsed_content.get('personnel_info', {}).get('defense_advocates', [])
        if personnel_advocates:
            return personnel_advocates

        # Look for advocates in speaker mapping
        advocates = []
        for speaker, speaker_type in parsed_content['speaker_mapping'].items():
            if speaker_type == 'defense':
                # Clean up advocate name
                clean_name = speaker.replace('_', ' ').replace('ADVOCATE ', '')
                advocate_name = f"Advocate {clean_name}"
                if advocate_name not in advocates:
                    advocates.append(advocate_name)

        return advocates if advocates else ["Defense advocates not identified"]

    def _extract_witnesses_refined(self, parsed_content: Dict) -> List[str]:
        """Refined witnesses extraction"""

        witnesses = []

        # Look for witnesses in speaker mapping
        for speaker, speaker_type in parsed_content['speaker_mapping'].items():
            if speaker_type == 'witness':
                # Skip procedural speakers
                if speaker not in ['PROCEEDINGS', 'CASE DETAILS', 'RELIEF SOUGHT', 'EVIDENCE']:
                    witness_name = speaker.replace('_', ' ')
                    # Add proper titles
                    if 'INSPECTOR' in witness_name:
                        witness_name = witness_name.replace('INSPECTOR ', 'Inspector ')
                    elif 'DR' in witness_name:
                        witness_name = witness_name.replace('DR ', 'Dr. ')
                    elif 'SMT' in witness_name:
                        witness_name = witness_name.replace('SMT ', 'Smt. ')

                    witnesses.append(witness_name)

        return witnesses if witnesses else ["Witnesses not identified"]

    def _extract_accused_refined(self, text: str, parsed_content: Dict) -> List[str]:
        """Refined accused persons extraction"""

        accused = []

        # Look in all relevant statements
        all_statements = (parsed_content.get('prosecutor_statements', []) +
                         parsed_content.get('defense_statements', []) +
                         parsed_content.get('witness_statements', []))

        patterns = [
            r'accused ([A-Za-z\s]+?)(?:\s|\.|\,)',
            r'Accused No\. \d+ \(([A-Za-z\s]+)\)',
            r'my client ([A-Za-z\s]+?)(?:\s|\.|\,)',
            r'defendant ([A-Za-z\s]+?)(?:\s|\.|\,)',
            r'respondent ([A-Za-z\s]+?)(?:\s|\.|\,)'
        ]

        for statement in all_statements:
            for pattern in patterns:
                matches = re.findall(pattern, statement, re.IGNORECASE)
                for match in matches:
                    accused_name = match.strip()
                    # Filter out common words and ensure meaningful names
                    if (len(accused_name) > 2 and
                        accused_name.lower() not in ['the', 'was', 'has', 'had', 'were', 'are'] and
                        accused_name not in accused):
                        accused.append(accused_name)

        return accused if accused else ["Accused not identified"]

    def _extract_complainant_refined(self, text: str, parsed_content: Dict) -> str:
        """Refined complainant extraction"""

        patterns = [
            r'complainant,? ([A-Za-z\s]+?)(?:\s|\.|\,)',
            r'Complainant:\s*([A-Za-z\s]+?)(?:\s|\.|\,)',
            r'complaint.*?([A-Za-z\s]+), father',
            r'written complaint from.*?([A-Za-z\s]+?)(?:\s|\.|\,)'
        ]

        # Check prosecutor statements first
        prosecutor_statements = parsed_content.get('prosecutor_statements', [])
        for statement in prosecutor_statements:
            for pattern in patterns:
                match = re.search(pattern, statement, re.IGNORECASE)
                if match:
                    name = match.group(1).strip()
                    if len(name) > 2:
                        return f"Shri {name}"

        # Fallback to full text
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if len(name) > 2:
                    return f"Shri {name}"

        return "Complainant not identified"

    def _extract_victim_refined(self, text: str, parsed_content: Dict) -> str:
        """Refined victim extraction"""

        patterns = [
            r'victim,? ([A-Za-z\s]+?)(?:\s|\.|\,)',
            r'deceased ([A-Za-z\s]+?)(?:\s|\.|\,)',
            r'Victim:\s*([A-Za-z\s]+?)(?:\s|\.|\,)',
            r'found ([A-Za-z\s]+?) (?:lying|dead)'
        ]

        # Check prosecutor and witness statements
        relevant_statements = (parsed_content.get('prosecutor_statements', []) +
                              parsed_content.get('witness_statements', []))

        for statement in relevant_statements:
            for pattern in patterns:
                match = re.search(pattern, statement, re.IGNORECASE)
                if match:
                    name = match.group(1).strip()
                    if len(name) > 2:
                        return f"Shri/Smt. {name}"

        return "Victim not identified"

    def _extract_fir_refined(self, text: str, parsed_content: Dict) -> Dict:
        """Refined FIR details extraction"""

        fir_info = {}

        # Look in prosecutor and witness statements
        relevant_statements = (parsed_content.get('prosecutor_statements', []) +
                              parsed_content.get('witness_statements', []))

        for statement in relevant_statements:
            # FIR number
            fir_match = re.search(r'FIR No\. (\d+/\d+)', statement)
            if fir_match and 'fir_number' not in fir_info:
                fir_info['fir_number'] = fir_match.group(1)

            # Sections
            sections_patterns = [
                r'registered under Sections ([^.]+?)(?:\s|\.)',
                r'under Sections ([^.]+?) (?:IPC|of)',
                r'Sections ([0-9, ]+) (?:IPC|and)'
            ]

            for pattern in sections_patterns:
                sections_match = re.search(pattern, statement)
                if sections_match and 'sections' not in fir_info:
                    sections = sections_match.group(1).strip()
                    # Clean up sections
                    sections = re.sub(r'\s+', ' ', sections)
                    fir_info['sections'] = sections
                    break

        return fir_info

    def _extract_police_station_refined(self, text: str, parsed_content: Dict) -> str:
        """Refined police station extraction"""

        patterns = [
            r'Police Station ([A-Za-z\s]+?)(?:\s|\.|\,)',
            r'PS ([A-Za-z\s]+?)(?:\s|\.|\,)',
            r'posted at ([A-Za-z\s]+?) Police Station',
            r'at PS ([A-Za-z\s]+?)(?:\s|\.|\,)'
        ]

        # Check witness statements (police officers)
        witness_statements = parsed_content.get('witness_statements', [])
        for statement in witness_statements:
            for pattern in patterns:
                match = re.search(pattern, statement, re.IGNORECASE)
                if match:
                    ps_name = match.group(1).strip()
                    if len(ps_name) > 2:
                        return ps_name

        return "Police Station not identified"

    def _extract_legal_sections_refined(self, text: str, parsed_content: Dict) -> Dict:
        """Refined legal sections extraction with better classification"""

        sections = {
            'ipc_sections': [],
            'crpc_sections': [],
            'evidence_act_sections': [],
            'other_sections': []
        }

        # Check all statements
        all_statements = (parsed_content.get('prosecutor_statements', []) +
                         parsed_content.get('defense_statements', []) +
                         parsed_content.get('witness_statements', []))

        # Known CrPC sections to avoid misclassification
        known_crpc_sections = ['161', '164', '125', '41', '154', '173', '207', '227', '228']

        for statement in all_statements:
            # IPC sections - explicit mention
            ipc_patterns = [
                r'Section (\d+[A-Za-z]*) (?:of the )?(?:Indian Penal Code|IPC)',
                r'Sections ([0-9, ]+) (?:of the )?(?:Indian Penal Code|IPC)',
                r'(\d+), (\d+), and (\d+) IPC'
            ]

            for pattern in ipc_patterns:
                matches = re.findall(pattern, statement, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        for section in match:
                            if section.strip() and section.strip() not in known_crpc_sections:
                                sections['ipc_sections'].append(section.strip())
                    else:
                        if ',' in match:
                            for section in match.split(','):
                                section = section.strip()
                                if section and section not in known_crpc_sections:
                                    sections['ipc_sections'].append(section)
                        else:
                            if match.strip() not in known_crpc_sections:
                                sections['ipc_sections'].append(match.strip())

            # CrPC sections
            crpc_patterns = [
                r'Section (\d+[A-Za-z]*) (?:of the )?(?:Criminal Procedure Code|CrPC|Cr\.P\.C\.)',
                r'under Section (\d+[A-Za-z]*) CrPC',
                r'interrogation under Section (\d+[A-Za-z]*)',
                r'statement under Section (\d+[A-Za-z]*)'
            ]

            for pattern in crpc_patterns:
                matches = re.findall(pattern, statement, re.IGNORECASE)
                sections['crpc_sections'].extend(matches)

            # Evidence Act sections
            evidence_patterns = [
                r'Section (\d+[A-Za-z]*) (?:of the )?Evidence Act',
                r'under Section (\d+[A-Za-z]*) Evidence Act'
            ]

            for pattern in evidence_patterns:
                matches = re.findall(pattern, statement, re.IGNORECASE)
                sections['evidence_act_sections'].extend(matches)

        # Remove duplicates and clean up
        for key in sections:
            sections[key] = list(set([s for s in sections[key] if s]))

        return sections

    def _extract_judge_content_refined(self, parsed_content: Dict) -> List[str]:
        """Refined judge observations extraction without truncation"""

        judge_statements = parsed_content.get('judge_statements', [])
        observations = []

        for statement in judge_statements:
            # Filter out very short procedural statements
            if len(statement) > 15:
                # Don't truncate - keep full content but limit number of statements
                observations.append(statement)

        return observations[:8] if observations else ["Judge observations not recorded"]

    def _extract_prosecutor_content_refined(self, parsed_content: Dict) -> List[str]:
        """Refined prosecutor arguments extraction without truncation"""

        prosecutor_statements = parsed_content.get('prosecutor_statements', [])
        arguments = []

        for statement in prosecutor_statements:
            if len(statement) > 20:
                # Keep full content without truncation
                arguments.append(statement)

        return arguments[:8] if arguments else ["Prosecutor arguments not recorded"]

    def _extract_defense_content_refined(self, parsed_content: Dict) -> List[str]:
        """Refined defense arguments extraction without truncation"""

        defense_statements = parsed_content.get('defense_statements', [])
        arguments = []

        for statement in defense_statements:
            if len(statement) > 20:
                # Keep full content without truncation
                arguments.append(statement)

        return arguments[:8] if arguments else ["Defense arguments not recorded"]

    def _extract_witness_content_refined(self, parsed_content: Dict) -> List[str]:
        """Refined witness testimony extraction without truncation"""

        witness_statements = parsed_content.get('witness_statements', [])
        testimony = []

        for statement in witness_statements:
            if len(statement) > 25:
                # Keep full content without truncation
                testimony.append(statement)

        return testimony[:8] if testimony else ["Witness testimony not recorded"]

    def _extract_case_law_refined(self, text: str, parsed_content: Dict) -> List[str]:
        """Refined case law extraction with better accuracy"""

        cases = []

        # Check defense statements (most likely to cite cases)
        defense_statements = parsed_content.get('defense_statements', [])

        patterns = [
            r'([A-Za-z\s\.]+) v\. ([A-Za-z\s\.]+), \((\d{4})\) (\d+) SCC (\d+)',
            r'in ([A-Za-z\s\.]+) v\. ([A-Za-z\s\.]+), \((\d{4})\)',
            r'case of ([A-Za-z\s\.]+) v\. ([A-Za-z\s\.]+), \((\d{4})\)'
        ]

        for statement in defense_statements:
            for pattern in patterns:
                matches = re.findall(pattern, statement)
                for match in matches:
                    if len(match) == 5:  # Full SCC citation
                        case_citation = f"{match[0].strip()} v. {match[1].strip()}, ({match[2]}) {match[3]} SCC {match[4]}"
                    elif len(match) == 3:  # Year only
                        case_citation = f"{match[0].strip()} v. {match[1].strip()} ({match[2]})"
                    else:
                        continue

                    # Filter out noise
                    if (len(match[0].strip()) > 3 and len(match[1].strip()) > 3 and
                        case_citation not in cases):
                        cases.append(case_citation)

        return cases if cases else ["No case law cited"]

    def _extract_evidence_refined(self, text: str, parsed_content: Dict) -> List[str]:
        """Refined evidence extraction with complete details"""

        evidence = []

        # Check prosecutor and witness statements
        relevant_statements = (parsed_content.get('prosecutor_statements', []) +
                              parsed_content.get('witness_statements', []))

        evidence_patterns = [
            r'recovered ([^.]+\.)',
            r'evidence ([^.]+\.)',
            r'CCTV ([^.]+\.)',
            r'forensic ([^.]+\.)',
            r'medical ([^.]+\.)',
            r'post-mortem ([^.]+\.)',
            r'blood-stained ([^.]+\.)',
            r'weapon ([^.]+\.)',
            r'fingerprint ([^.]+\.)'
        ]

        for statement in relevant_statements:
            for pattern in evidence_patterns:
                matches = re.findall(pattern, statement, re.IGNORECASE)
                for match in matches:
                    evidence_item = match.strip()
                    if len(evidence_item) > 10 and evidence_item not in evidence:
                        evidence.append(evidence_item)

        return evidence[:8] if evidence else ["Evidence details not specified"]

    def _extract_key_facts_refined(self, text: str, parsed_content: Dict) -> List[str]:
        """Refined key facts extraction with complete information"""

        facts = []

        # Check prosecutor statements for key facts
        prosecutor_statements = parsed_content.get('prosecutor_statements', [])

        fact_patterns = [
            r'On ([^.]+\.)',
            r'During investigation ([^.]+\.)',
            r'The deceased ([^.]+\.)',
            r'The complainant ([^.]+\.)',
            r'we received ([^.]+\.)',
            r'found that ([^.]+\.)'
        ]

        for statement in prosecutor_statements:
            for pattern in fact_patterns:
                matches = re.findall(pattern, statement, re.IGNORECASE)
                for match in matches:
                    fact = match.strip()
                    if len(fact) > 15 and fact not in facts:
                        facts.append(fact)

        return facts[:8] if facts else ["Key facts not identified"]

    def _extract_next_date_refined(self, text: str) -> str:
        """Refined next hearing date extraction"""

        patterns = [
            r'adjourned to ([^\\n]+?) for',
            r'adjourned to ([^.]+\.)',
            r'resume.*?(\d{1,2}[a-z]{2}\s+[A-Za-z]+,?\s+\d{4})',
            r'posted.*?(\d{1,2}[a-z]{2}\s+[A-Za-z]+,?\s+\d{4})',
            r'(\d{1,2}[a-z]{2}\s+[A-Za-z]+,?\s+\d{4}).*?for'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                next_date = match.group(1).strip()
                # Clean up the date
                next_date = re.sub(r'\s+', ' ', next_date)
                return next_date

        return "Next date not specified"

    def _extract_judgment_status_refined(self, text: str) -> str:
        """Refined judgment status extraction"""

        status_indicators = [
            ('judgment.*?reserved', 'Judgment Reserved'),
            ('arguments.*?concluded', 'Arguments Concluded'),
            ('matter.*?heard', 'Arguments Heard'),
            ('court.*?adjourned', 'Proceedings Adjourned'),
            ('examination.*?witnesses', 'Witness Examination Ongoing'),
            ('recording.*?evidence', 'Evidence Recording Ongoing'),
            ('framing.*?issues', 'Issues Framing Stage'),
            ('final.*?arguments', 'Final Arguments Stage')
        ]

        text_lower = text.lower()
        for indicator, status in status_indicators:
            if re.search(indicator, text_lower):
                return status

        return "Proceedings ongoing"

# -----------------------------------------------------------------------------
# Refined Summary Generator
# -----------------------------------------------------------------------------
class RefinedSummaryGenerator:
    """Refined summary generator with complete content and better formatting"""

    def __init__(self, case_type: str):
        self.case_type = case_type

    def generate_refined_summary(self, case_info: Dict, original_text: str) -> str:
        """Generate comprehensive refined summary"""

        # Generate enhanced narrative summary
        narrative_summary = self._generate_refined_narrative(case_info)

        # Generate enhanced structured analysis
        structured_analysis = self._generate_refined_structured_analysis(case_info)

        # Combine in professional format
        summary = f"""{'='*100}
REFINED DIARIZED INDIAN LEGAL CASE ANALYSIS - PRODUCTION VERSION
{'='*100}

EXECUTIVE SUMMARY (कार्यकारी सारांश):
{'-'*60}
{narrative_summary}

{'='*100}
DETAILED STRUCTURED ANALYSIS (विस्तृत संरचित विश्लेषण)
{'='*100}
{structured_analysis}

{'='*100}
END OF ANALYSIS (विश्लेषण समाप्त)
{'='*100}"""

        return summary

    def _generate_refined_narrative(self, case_info: Dict) -> str:
        """Generate refined narrative summary with complete information"""

        narrative_parts = []

        # Case introduction with complete details
        case_name = case_info.get('case_name', '')
        case_number = case_info.get('case_number', '')
        court_name = case_info.get('court_name', '')
        judge_name = case_info.get('judge_name', '')
        hearing_date = case_info.get('hearing_date', '')

        if case_name and 'not identified' not in case_name:
            intro = f"In the matter of {case_name}"
            if case_number and 'not identified' not in case_number:
                intro += f" ({case_number})"
            narrative_parts.append(intro)

        if court_name and 'not identified' not in court_name:
            court_info = f"before the {court_name}"
            if judge_name and 'not identified' not in judge_name:
                court_info += f", presided over by Hon'ble {judge_name}"
            if hearing_date and 'not specified' not in hearing_date:
                court_info += f" on {hearing_date}"
            narrative_parts.append(court_info)

        # Personnel information
        prosecutor = case_info.get('public_prosecutor', '')
        defense_advocates = case_info.get('defense_advocates', [])

        if prosecutor and 'not identified' not in prosecutor:
            narrative_parts.append(f"The case was argued by {prosecutor} for the State")

        if defense_advocates and 'not identified' not in str(defense_advocates):
            if len(defense_advocates) == 1:
                narrative_parts.append(f"while the accused was represented by {defense_advocates[0]}")
            else:
                advocates_str = ', '.join(defense_advocates)
                narrative_parts.append(f"while the accused were represented by {advocates_str}")

        # Key proceedings summary
        key_facts = case_info.get('key_facts', [])
        if key_facts and 'not identified' not in str(key_facts):
            narrative_parts.append(f"The case involves key facts including {key_facts[0]}")

        # Evidence summary
        evidence = case_info.get('evidence_presented', [])
        if evidence and 'not specified' not in str(evidence):
            narrative_parts.append(f"Evidence presented included {evidence[0]}")

        # Outcome
        judgment_status = case_info.get('judgment_status', '')
        next_date = case_info.get('next_date', '')

        if judgment_status and 'ongoing' not in judgment_status:
            outcome = f"The proceedings concluded with {judgment_status.lower()}"
            if next_date and 'not specified' not in next_date:
                outcome += f" and the matter was adjourned to {next_date}"
            narrative_parts.append(outcome)

        return ". ".join(narrative_parts) + "." if narrative_parts else "Case proceedings summary not available."

    def _generate_refined_structured_analysis(self, case_info: Dict) -> str:
        """Generate refined structured analysis with complete information"""

        structure = f"""CASE IDENTIFICATION (मामले की पहचान):
Case Name: {case_info.get('case_name', 'Not identified')}
Case Number: {case_info.get('case_number', 'Not identified')}
Case Type: {case_info.get('case_type', 'Not specified').title()}
Court: {case_info.get('court_name', 'Not identified')}
Judge: {case_info.get('judge_name', 'Not identified')}
Hearing Date: {case_info.get('hearing_date', 'Not specified')}

LEGAL PERSONNEL (कानूनी कर्मचारी):
Public Prosecutor: {case_info.get('public_prosecutor', 'Not identified')}
Defense Advocates:
{self._format_list_refined(case_info.get('defense_advocates', ['Not identified']))}
Witnesses Present:
{self._format_list_refined(case_info.get('witnesses', ['Not identified']))}

PARTIES (पक्षकार):
Accused Persons:
{self._format_list_refined(case_info.get('accused_persons', ['Not identified']))}
Complainant: {case_info.get('complainant', 'Not identified')}
Victim: {case_info.get('victim', 'Not identified')}

FIR DETAILS (एफआईआर विवरण):
{self._format_fir_details_refined(case_info.get('fir_details', {}))}
Police Station: {case_info.get('police_station', 'Not identified')}

LEGAL PROVISIONS (कानूनी प्रावधान):
{self._format_legal_sections_refined(case_info.get('legal_sections', {}))}

COURT PROCEEDINGS (न्यायालय की कार्यवाही):
Judge Observations:
{self._format_list_refined(case_info.get('judge_observations', ['Not recorded']))}

Prosecutor Arguments:
{self._format_list_refined(case_info.get('prosecutor_arguments', ['Not recorded']))}

Defense Arguments:
{self._format_list_refined(case_info.get('defense_arguments', ['Not recorded']))}

Witness Testimony:
{self._format_list_refined(case_info.get('witness_testimony', ['Not recorded']))}

EVIDENCE PRESENTED (प्रस्तुत साक्ष्य):
{self._format_list_refined(case_info.get('evidence_presented', ['Not specified']))}

KEY FACTS (मुख्य तथ्य):
{self._format_list_refined(case_info.get('key_facts', ['Not identified']))}

CASE LAW CITED (उद्धृत मामला कानून):
{self._format_list_refined(case_info.get('case_law_cited', ['No case law cited']))}

JUDGMENT STATUS (निर्णय की स्थिति):
Status: {case_info.get('judgment_status', 'Proceedings ongoing')}
Next Date: {case_info.get('next_date', 'Not specified')}"""

        return structure

    def _format_list_refined(self, items: List[str]) -> str:
        """Refined list formatting with complete content"""
        if not items or (len(items) == 1 and any(phrase in items[0].lower() for phrase in ['not', 'none'])):
            return "• Not specified"

        # Filter out generic responses but keep meaningful content
        meaningful_items = []
        for item in items:
            if item and not any(phrase in item.lower() for phrase in
                              ['not specified', 'not identified', 'not recorded', 'none specified']):
                meaningful_items.append(item)

        if not meaningful_items:
            return "• Not specified"

        # Format with proper bullet points and preserve full content
        formatted_items = []
        for item in meaningful_items:
            # Don't truncate - keep full content
            formatted_items.append(f"• {item}")

        return '\n'.join(formatted_items)

    def _format_fir_details_refined(self, fir_details: Dict) -> str:
        """Refined FIR details formatting"""
        if not fir_details:
            return "• FIR details not available"

        details = []
        if fir_details.get('fir_number'):
            details.append(f"• FIR Number: {fir_details['fir_number']}")
        if fir_details.get('sections'):
            details.append(f"• Sections: {fir_details['sections']}")

        return '\n'.join(details) if details else "• FIR details not available"

    def _format_legal_sections_refined(self, sections: Dict) -> str:
        """Refined legal sections formatting"""
        if not sections:
            return "• Legal sections not specified"

        formatted = []
        if sections.get('ipc_sections'):
            ipc_list = ', '.join(sections['ipc_sections'])
            formatted.append(f"IPC Sections: {ipc_list}")
        if sections.get('crpc_sections'):
            crpc_list = ', '.join(sections['crpc_sections'])
            formatted.append(f"CrPC Sections: {crpc_list}")
        if sections.get('evidence_act_sections'):
            evidence_list = ', '.join(sections['evidence_act_sections'])
            formatted.append(f"Evidence Act Sections: {evidence_list}")

        return '\n'.join([f"• {item}" for item in formatted]) if formatted else "• Legal sections not specified"

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main():
    logger.info("Starting Refined Diarized Indian Legal Summarizer...")

    # Read input
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        logger.error(f"Could not read input file: {e}")
        return

    logger.info(f"Input text length: {len(text)} characters")

    # Enhanced case type detection
    case_type = 'criminal'  # Default
    text_lower = text.lower()

    if any(term in text_lower for term in ['civil suit', 'plaintiff', 'defendant', 'damages', 'contract']):
        case_type = 'civil'
    elif any(term in text_lower for term in ['matrimonial', 'divorce', 'family court', 'custody', 'maintenance']):
        case_type = 'family'
    elif any(term in text_lower for term in ['writ petition', 'mandamus', 'fundamental rights', 'constitutional']):
        case_type = 'constitutional'

    logger.info(f"Detected case type: {case_type}")

    # Extract information with refined methods
    extractor = RefinedDiarizedExtractor(case_type)
    case_info = extractor.extract_refined_case_info(text)

    # Generate refined summary
    generator = RefinedSummaryGenerator(case_type)
    summary = generator.generate_refined_summary(case_info, text)

    # Write output
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(summary)
    except Exception as e:
        logger.error(f"Could not write output file: {e}")
        return

    logger.info(f"Refined summary written to {args.output}")
    logger.info(f"Summary length: {len(summary)} characters")

    print(summary)

if __name__ == "__main__":
    main()
