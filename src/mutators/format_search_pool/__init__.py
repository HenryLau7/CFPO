from .query_format import QA_query_format_pool, query_format_desc_map_QA, generated_query_format_pool_QA,  generated_query_format_pool_MultiChoice, multiple_choice_query_format_pool, query_format_desc_map_MultiChoice, classification_query_format_pool
from .prompt_renderer import prompt_renderer_pool, prompt_format_desc_map, generated_prompt_renderer_pool

SEARCH_POOL ={
    "QA":{
        "query":QA_query_format_pool,
        "prompt":prompt_renderer_pool,
        "generated_query":generated_query_format_pool_QA,
        "generated_prompt":generated_prompt_renderer_pool,
        "query_desc": query_format_desc_map_QA,
        "prompt_desc": prompt_format_desc_map
    },
    "Classfication":
    {
        "query":classification_query_format_pool,
        "prompt":prompt_renderer_pool,
        "generated_query":generated_query_format_pool_QA,
        "generated_prompt":generated_prompt_renderer_pool,
        "query_desc": query_format_desc_map_QA,
        "prompt_desc": prompt_format_desc_map
    },
    "MultiChoice":{
        "query":multiple_choice_query_format_pool,
        "prompt":prompt_renderer_pool,
        "generated_query":generated_query_format_pool_MultiChoice,
        "generated_prompt":generated_prompt_renderer_pool,
        "query_desc": query_format_desc_map_MultiChoice,
        "prompt_desc": prompt_format_desc_map
    }
}
