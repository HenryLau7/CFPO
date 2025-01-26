from .markdown import *
from .html import *
from .xml import *
from .latex import *
from .json import *
from .plain import *
from .direct_joint import *
from .generated_format import *

prompt_renderer_pool = [
    (direct_joint_renderer, direct_joint_extractor),
    (markdown_renderer, markdown_extractor),
    (plain_renderer, plain_extractor),
    (html_renderer, html_extractor),
    (xml_renderer, xml_extractor),
    (latex_renderer, latex_extractor),
    (json_renderer, json_extractor)
]

prompt_format_desc_map = {
    direct_joint_renderer: "This renderer constructs a prompt by sequentially concatenating non-empty strings from the provided parameters: task_instruction, task_detail, output_format, examples, and query_part. Each parameter is added with a double newline separator, ensuring a clear division between different sections of the generated prompt.",
    plain_renderer: "This renderer creates a prompt by formatting the input parameters—task_instruction, task_detail, output_format, examples, and query_part—into an ordered dictionary and then converting this dictionary into a string. Each key (like \"Task Instruction\") is followed by its corresponding content, separated by a colon and newline. The 'examples' and 'query_part' are concatenated with a double newline in between. The output is a cleanly formatted prompt where each section is clearly labeled and separated, enhancing readability and organization.",
    markdown_renderer: "This renderer assembles a prompt by merging the given parameters—task_instruction, task_detail, output_format, examples, and query_part—into a markdown-formatted string. Each parameter is associated with a section name and formatted with markdown headers (using \"#####\") for clarity and emphasis. The 'examples' and 'query_part' are combined with a double newline, creating a distinct \"Examples\" section. This method results in a prompt that is not only well-organized but also visually segmented, enhancing the user's ability to navigate different parts of the text easily.",
    xml_renderer: "This renderer constructs an XML-formatted prompt by first grouping the provided parameters—task_instruction, task_detail, output_format, examples, and query_part—into an ordered dictionary. Each key-value pair is then transformed into an XML element where the key becomes the tag and the content is nested inside. Special handling is applied to the \"Examples\" section, where the closing tag is removed to maintain a particular formatting style. The result is a structured XML document that maintains a clear and hierarchical representation of the prompt components, allowing for easy parsing and manipulation in environments that support XML processing.",
    html_renderer: "This renderer constructs an HTML-formatted prompt by organizing the provided parameters—task_instruction, task_detail, output_format, examples, and query_part—into an ordered dictionary. It then translates each key-value pair into HTML elements, with each parameter encapsulated within a <div> tag marked by a class that corresponds to its name. Headers (<h2>) are used for labeling sections, and paragraph tags (<p>) for content. The \"Examples\" section is formatted slightly differently, maintaining an open <p> tag, which suggests a continuation or additional content might follow. This method produces a well-structured HTML document that is easy to read and visually appealing, facilitating better engagement and clarity for users viewing the rendered content in a web browser.",
    latex_renderer: "This renderer constructs a LaTeX-formatted prompt by organizing the provided parameters—task_instruction, task_detail, output_format, examples, and query_part—into distinct sections using LaTeX commands. Each parameter is rendered under its respective section using the LaTeX `\\section\{\}` command for headers, with the content following each header. The output is a well-structured LaTeX document that facilitates easy rendering and typesetting, making it ideal for academic or formal document preparation.",
    json_renderer: "This renderer constructs a JSON-formatted prompt by organizing the provided parameters—task_instruction, task_detail, output_format, examples, and query_part—into a structured dictionary. Each parameter is assigned to a corresponding key in the dictionary. The 'Examples' key combines the 'examples' and 'query_part' parameters, separated by a double newline for clarity. The resulting dictionary is then serialized into a well-indented JSON string using the `json.dumps` method, making it machine-readable and easily integrable into systems that require structured data. This approach ensures that the prompt is not only human-readable but also easily parsed and utilized in automated workflows.",
}