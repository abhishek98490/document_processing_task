import os
from dotenv import load_dotenv
import sys
from src.logging import logging
import openai

load_dotenv()

class LLM():
    
    def __init__(self):
        
        self.client = openai.OpenAI(
                    api_key=os.environ.get("SAMBANOVA_API_KEY"),
                    base_url="https://api.sambanova.ai/v1",
                )
        
    def get_prompt(self, context: str, query: str):

        prompt = f"""Based on the following context, 
                    please provide a relevant and contextual response. If the answer cannot 
                    be derived from the context, say "I cannot answer this based on the provided information."

                    Context from documents:
                    {context}

                    Human: {query}

                    Assistant:"""

        return prompt
    
    def chat(self,context,query):
        try:
            response = self.client.chat.completions.create(
                model="Meta-Llama-3.1-8B-Instruct",
                messages=[
                            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                            {"role": "user", "content": self.get_prompt(context,query)}
                        ],
                max_tokens = 150
                )
            logging.info("Response generated")
            return response.choices[0].message.content

        except Exception as e:
            logging.info("Error occured while Generating the output")
            return e
    
if __name__=="__main__":    
    LLM = LLM()
    context = """to intensify in the coming decades. These effects include: Rising Temperatures Global temperatures have risen by about 1.2 degrees Celsius (2.2 degrees Fahrenheit) since the late 19th century. This warming is not uniform, with some regions experiencing more significant increases than others. Heatwaves Heatwaves are becoming more frequent and severe, posing risks to human health, agriculture, and infrastructure. Cities are particularly vulnerable due to the "urban heat island" effect. Heatwaves can lead to heat -related illnesses and exacerbate existing h ealth conditions. Changing Seasons Climate change is altering the timing and length of seasons, affecting ecosystems and human activities. For example, spring is arriving earlier, and winters are becoming shorter and milder in many regions. This shift disrupts plant and animal life cycles a nd agricultural practices. Melting Ice and Rising Sea Levels Warmer temperatures are causing polar ice caps and glaciers to melt, contributing to rising sea levels. Sea levels have risen by about 20 centimeters (8 inches) in the past century, threatening coastal communities and ecosystems. Polar Ice Melt The Arctic is warming at more than twice the global average rate, leading to significant ice loss. Antarctic ice sheets are also losing mass, contributing to sea level rise.
    Understanding Climate Change Chapter 1: Introduction to Climate Change Climate change refers to significant, long -term changes in the global climate. The term "global climate" encompasses the planet's overall weather patterns, including temperature, precipitation, and wind patterns, over an extended period. Over the past cent ury, human activities, particularly the burning of fossil fuels and deforestation, have significantly contributed to climate change. Historical Context The Earth's climate has changed throughout history. Over the past 650,000 years, there have been seven cycles of glacial advance and retreat, with the abrupt end of the last ice age about 11,700 years ago marking the beginning of the modern climate era and human civilization. Most of these climate changes are attributed to very small variations in Earth's orbit that change the amount of solar energy our planet receives. During the Holocene epoch, which began at the end of the last ice age, human societies f lourished, but the industrial era has seen unprecedented changes. Modern Observations Modern scientific observations indicate a rapid increase in global temperatures, sea levels, and extreme weather events. The Intergovernmental Panel on Climate Change (IPCC) has documented these changes extensively. Ice core samples, tree rings, and ocean sediments provide a"""
    query = "How much degree celsius does global temprature has risen?"
    print(LLM.chat(context,query))