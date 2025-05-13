import json

class CodeExtractor:
    def __init__(self, llm, logger):

        self.logger = logger
        self.llm = llm

    async def extract_chapters(self, response, tags=[]):
        """
        Extracts the HTS chapters from the response.

        Args:
            response (str): Contains the response from the LLM.

        Returns:
            list[str]: The extracted HTS chapters.
        """
        prompt = f"""Review the response and extract the HTS Chapters into a list. 
HTS Chapters are two digit numbers, such as 01, 02, 03, etc. 
Extract only the final Chapters selected in the response, and do not include any general reference to chapters such "I consulted chapters 10-20" etc.
Here is a response selecting 3 chapters, please extract the chapters to a list: {response}"""
        
        output_structure = {
            'type': 'OBJECT',
            'properties': {
                'chapters_list': {'type': 'ARRAY', 'items': {'type': 'STRING'}},
            },
            'required': ['chapters_list'],
        }

        try:
            output = await self.llm.ainvoke(prompt,
            generation_config={"response_mime_type":'application/json',
            "response_schema": output_structure})
            
            chapters = json.loads(output.content)

            self.logger.info(f"Extracted chapters list: {chapters['chapters_list']}", _tags=tags)
            
            return chapters['chapters_list']
        except Exception as e:
            self.logger.error(f"Error extracting chapters: {e}")
            raise e
        
    async def extract_four_digit_codes(self, response, tags=[]):
        """
        Extracts the HTS four-digit codes from the response.

        Args:
            response (str): Contains the response from the LLM.

        Returns:
            list[str]: The extracted HTS four-digit codes.
        """
        prompt = f"""Review the response and extract the four digit HTS Codes into a list. 
HTS Codes are four digit numbers, such as 0101, 2345, 0390, etc. 
Extract only the final codes selected in the response, and do not include any general reference to codes such "I consulted codes 1000-9000" etc.
Here is a response selecting up to 6 codes, please extract the codes to a list: {response}"""

        output_structure = {
            'type': 'OBJECT',
            'properties': {
                'code_list': {'type': 'ARRAY', 'items': {'type': 'STRING'}},
            },
            'required': ['code_list'],
        }
        
        try:
            output = await self.llm.ainvoke(prompt,
            generation_config={"response_mime_type":'application/json',
            "response_schema": output_structure})
            
            codes = json.loads(output.content)

            self.logger.info(f"Extracted four-digit code list: {codes['code_list']}", _tags=tags)
            
            return codes['code_list']
        except Exception as e:
            self.logger.error(f"Error extracting four-digit codes: {e}")
            raise e
    
    async def extract_full_codes(self, response, tags=[]):
        """
        Extracts the full HTS codes from the response.

        Args:
            response (str): Contains the response from the LLM.

        Returns:
            list[str]: The extracted HTS full codes.
        """
        prompt = f"""Review the response and extract the full HTS Codes into a list. 
HTS Codes are full codes, such as 0101.01.9029, 2345.02.9800, 0390.03.3545, etc. 
Extract only the final codes selected in the response, and do not include any general reference to codes such "I consulted codes 1000-9000" etc.
Here is a response selecting up to 6 codes, please extract the codes to a list: {response}"""
        
        output_structure = {
            'type': 'OBJECT',
            'properties': {
                'code_list': {'type': 'ARRAY', 'items': {'type': 'STRING'}},
            },
            'required': ['code_list'],
        }
        
        try:
            response = await self.llm.ainvoke(prompt,
            generation_config={"response_mime_type":'application/json',
            "response_schema": output_structure})
            
            codes = json.loads(response.content)

            self.logger.info(f"Extracted full code list: {codes['code_list']}", _tags=tags)
            
            return codes['code_list']
        except Exception as e:
            self.logger.error(f"Error extracting full codes: {e}")
            raise e