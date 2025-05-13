import json
import re
from dotenv import load_dotenv
load_dotenv()

from firecrawl import FirecrawlApp
from pathlib import Path
import pandas as pd
import aiohttp
import asyncio

class AgentActions:

    def __init__(self, logger, chapter_descs, four_digit_codes, final_full_codes, tariffy_org_id, tariffy_api_key, simpleduty_api_key):
        self.logger = logger
        self.chapter_descs = chapter_descs
        self.four_digit_codes = four_digit_codes
        self.final_full_codes = final_full_codes
        self.tariffy_org_id = tariffy_org_id
        self.tariffy_api_key = tariffy_api_key
        self.simpleduty_api_key = simpleduty_api_key
    
    @staticmethod
    def get_hts_headers(app: FirecrawlApp, headers_save_path: Path = 'chapter_headers_final.txt', chapter_desc_save_path: Path = 'chapter_desc.json') -> list:
        # Scrape the usitc website for HTS headers:
        scrape_result = app.scrape_url('https://hts.usitc.gov/', params={'formats': ['markdown']})

        # clean and extract the headers from the markdown content
        headers = scrape_result['markdown'].split('\n')
        headers = [line.strip() for line in headers if line.strip()] 
        headers = [re.sub(r'\(.*?\)', '', header).strip() for header in headers]
        headers = [i for i in headers if i not in ('Export','[Download]')]
        start_index = headers.index('General Statistical Notes')
        end_index = headers.index('- Section XXII:')
        headers = headers[start_index+1:end_index]

        chapter_descs = {}
        for i, line in enumerate(headers):
            line = line.strip()
            if line.startswith('- ### [Chapter ') and line.endswith(']'):
                # Extract the chapter number
                chapter_number = line.split('[Chapter ')[1].split(']')[0]
                if len(chapter_number) < 2:
                    chapter_number = '0' + chapter_number
                # Get the description from the next line
                if i + 1 < len(headers):
                    description = headers[i + 1].strip()
                    chapter_descs[chapter_number] = description
        
        # save headers to a text file
        with open(headers_save_path, 'w', encoding='utf-8') as file:
            file.write('\n'.join(headers))

        with open(chapter_desc_save_path, 'w', encoding='utf-8') as json_file:
            json.dump(chapter_descs, json_file, ensure_ascii=False)
        
        return headers, chapter_descs
    
    @staticmethod
    def wrangle_hts_data(htsdata: list) -> tuple[list[dict], list[dict]]:
        """
        Process raw HTS data to enrich descriptions and extract relevant code levels.
        
        Args:
            htsdata: Raw HTS data list containing dictionaries with htsno, description, indent, etc.
            
        Returns:
            tuple: (four_digit_codes, final_full_codes) where:
                - four_digit_codes: List of dicts with 4-digit HTS codes and descriptions
                - final_full_codes: List of dicts with 13-digit HTS codes, descriptions, and duty rates
        """
        # Filter out chapters 98 and 99
        data_step_1 = [i for i in htsdata if i['htsno'][:2] not in ('98', '99')]
        
        # Enhance descriptions for indented items (section headers)
        description_to_append = None
        level = None
        for i in data_step_1:
            if level is not None:
                if int(level) >= int(i['indent']):
                    level = None
                elif int(i['indent']) == int(level) + 1:
                    i['description'] = description_to_append + "<br> " + i['description']
            if i['htsno'] == '':
                description_to_append = i['description']
                level = i['indent']
        
        # Filter out items without HTS numbers
        data_step_2 = [i for i in data_step_1 if i['htsno'] != '']
        
        # Extract four digit codes for initial classification
        four_digit_codes = [
            {'htsno': i['htsno'], 'description': i['description']} 
            for i in data_step_2 if len(i['htsno']) == 4
        ]
        
        # Get sorted unique indent levels
        levels = sorted(list(set(int(i['indent']) for i in data_step_2)))
        levels = [str(i) for i in levels]
        
        # Propagate description information down the hierarchy
        for level in levels:
            htsno_to_check = None
            description_to_append = None
            appended_to = set()
            
            for i in data_step_2:
                if level == i['indent']:
                    htsno_to_check = i['htsno']
                    description_to_append = i['description']
                    appended_to = set()
                
                if (htsno_to_check and int(i['indent']) > int(level) and 
                        htsno_to_check in i['htsno']):
                    prefix = i['htsno'][:len(htsno_to_check) + 3]
                    if prefix not in appended_to:
                        i['description'] = description_to_append + '<br> ' + i['description']
                        appended_to.add(prefix)
        
        # Propagate duty rates down the hierarchy
        htsno_to_check = None
        duty_to_append = None
        for i in data_step_2:
            if i['general'] != '':
                duty_to_append = i['general']
                htsno_to_check = i['htsno']
            elif htsno_to_check and htsno_to_check in i['htsno'] and i['general'] == '':
                i['general'] = duty_to_append
        
        # Extract final full codes (13-digit)
        final_full_codes = [
            {'htsno': i['htsno'], 'description': i['description'], 'duty_rate': i['general']} 
            for i in data_step_2 if len(i['htsno']) == 13
        ]
        
        return four_digit_codes, final_full_codes

    async def get_tariffy_codes(self, descriptions: list) -> list[dict]:
        """
        Get HTS codes from the Tariffy API based on product descriptions.
        """
        url = "https://api.tariffy.net/v1/lookup-codes"
        headers = {"Content-Type": "application/json"}
        data = {
            "organization_id": self.tariffy_org_id,
            "api_key": self.tariffy_api_key,
            "descriptions": descriptions,
            "region": "usa",
            "language": "en"
        }

        try:
            with self.logger.span('Calling Tariffy API', _level='info'):
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=data) as response:
                        if response.status == 200:
                            response_data = await response.json()
                            return [{'description': item['description'], 'tariffy_hts_code': item['hs_code_usa']} for item in response_data]
                        else:
                            raise Exception(f"API call failed with status code {response.status}")
        except Exception as e:
            # Log the error and return a fallback response
            fallback_response = [{"description": desc, "tariffy_hts_code": "unable to retrieve code"} for desc in descriptions]
            return fallback_response
    
    async def get_duty_rates(self, origin: str, dest: str, code: str) -> list[dict]:
     
        formatted_code = re.sub(r'\.', '', code)
        formatted_code = f"{formatted_code[:4]}.{formatted_code[4:6]}.{formatted_code[6:]}"

        url = "https://www.api.simplyduty.com/api/duty/getduty"  # API endpoint [cite: 51]

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.simpleduty_api_key  
        }

        payload = json.dumps({
            "HSCode": formatted_code,
            "OriginCountryCode": origin,  
            "DestinationCountryCode": dest,  
            "TempTariff": True,
        })

        try:
            with self.logger.span('Calling SimplyDuty API', _level='debug'):
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, data=payload) as response:
                        if response.status == 200:
                            response_data = await response.json()
                            return {'code': code, 'DutyRate': response_data['duty']['DutyRate']}
                        else:
                            raise Exception(f"API call failed with status code {response.status}")
        except Exception as e:
            # Log the error and return a fallback response
            fallback_response = {"code": code, "DutyRate": "unable to retrieve code"}
            return fallback_response
    
    async def get_rates_and_descs(self, origin: str, dest: str, code_one: str, code_two: str, code_three: str) -> dict:
        """
        Get the duty rates and descriptions for a given HTS code.
        
        Args:
            origin (str): The country of origin.
            dest (str): The destination country.
            code_one (str): The first HTS code to look up.
            code_two (str): The second HTS code to look up.
            code_three (str): The third HTS code to look up.
        
        Returns:
            dict: A dictionary with the HTS code, description, and duty rate.
        """
        unique_codes = {code_one, code_two, code_three}
        
        duty_tasks = [
            self.get_duty_rates(origin, dest, code) 
            for code in unique_codes
        ]

        duty_results = await asyncio.gather(*duty_tasks)
        duty_rates = {result['code']: result['DutyRate'] for result in duty_results}
        description_results = self.get_code_descriptions(list(unique_codes))
        descriptions = {result['code']: result['description'] for result in description_results}

        response_dict = {
        "most_likely_code_desc": descriptions[code_one],
        "most_likely_code_duty_rate": duty_rates[code_one],
        "most_likely_code_lower_rate_desc": descriptions[code_two],
        "most_likely_code_lower_rate_duty_rate": duty_rates[code_two],
        "tariffy_hts_code_desc": descriptions[code_three],
        "tariffy_hts_code_duty_rate": duty_rates[code_three]
        }
        return response_dict


    def get_four_digit_code_options(self, chapter_list, tags=[]) -> str:
        """
        Get the relevant level 1 data based on the chapter list.
        """
        four_digit_code_options = []
        try:
            for chapter in chapter_list:
                for item in self.four_digit_codes:
                    if item['htsno'][:2] == chapter:
                        four_digit_code_options.append(item)
            self.logger.info(f"Relevant four-digit codes: {four_digit_code_options[0:10]} .etc ..", _tags=tags)
        except Exception as e:
            self.logger.exception(f"Error getting four-digit code options: {e}")
            raise e
        return json.dumps(four_digit_code_options)

    def get_full_code_options(self, codes, four_digits=True, tags=[]) -> str:
        """
        Get the relevant full code options based on the four-digit code list.
        """
        full_code_options = []
        try:
            for code in codes:
                for item in self.final_full_codes:
                    if four_digits:
                        if item['htsno'][:4] == code:
                            full_code_options.append(item)
                    else:
                        if item['htsno'] == code:
                            full_code_options.append(item)
            self.logger.info(f"Relevant full codes: {full_code_options[0:10]} .etc ..", _tags=tags)
        except Exception as e:
            self.logger.exception(f"Error getting full code options: {e}")
            raise e
        return json.dumps(full_code_options)
    
    def get_code_descriptions(self, codes: list) -> pd.DataFrame:
        """
        Get the descriptions of a list of HTS codes and return as a DataFrame.
        
        Args:
            codes (list): A list of HTS codes to look up.
        
        Returns:
            pd.DataFrame: A DataFrame with columns 'code' and 'description'.
        """

        data = []
        try:
            for code in codes:
                description = next((self.chapter_descs[code[0:2]] + ":<br>" + item['description'] for item in self.final_full_codes if item['htsno'] == code), "Code not found")
                data.append({'code': code, 'description': description})
        except Exception as e:
            self.logger.exception(f"Error getting code descriptions: {e}")
            raise e
        
        return data

