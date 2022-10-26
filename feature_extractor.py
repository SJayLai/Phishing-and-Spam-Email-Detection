from url_features import *
from urllib.parse import urlparse
import tldextract
import urllib.parse

def get_domain(url):
    o = urllib.parse.urlsplit(url)
    return o.hostname, tldextract.extract(url).domain, o.path

def feature_extract_from_url(url):

    def words_raw_extraction(domain, subdomain, path):
        w_domain = re.split("\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", domain.lower())
        w_subdomain = re.split("\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", subdomain.lower())   
        w_path = re.split("\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", path.lower())
        raw_words = w_domain + w_path + w_subdomain
        w_host = w_domain + w_subdomain
        raw_words = list(filter(None,raw_words))
        return raw_words, list(filter(None,w_host)), list(filter(None,w_path))


    hostname, second_level_domain, path = get_domain(url)
    extracted_domain = tldextract.extract(url)
    domain_name = extracted_domain.domain + '.' + extracted_domain.suffix
    subdomain = extracted_domain.subdomain
    words_raw, words_raw_host, words_raw_path = words_raw_extraction(extracted_domain.domain, subdomain, path[1:])
    
    # print(path)
    # tmp = url[url.find(extracted_domain.suffix):len(url)]
    # pth = tmp.partition("/")
    # print(pth)
    # #path[1] = /, path[2] = path
    # path = pth[1] + pth[2]
    # print(pth[1], pth[2])


    output = [
        getLength(url),
        count_at(url),
        count_comma(url),
        count_dollar(url),
        count_semicolumn(url),
        count_space(url),
        count_and(url),
        count_slash(url),
        count_equal(url),
        count_percentage(url),
        count_exclamation(url),
        count_underscore(url),
        count_hyphens(url),
        
        count_colon(url),
        count_star(url),
        count_or(url),
        count_subdomain(url),
        prefix_suffix(url),
        port(url),
        punycode(url),
        count_tilde(url),
        having_ip_address(url),

        path_extension(path),
        count_http_token(path),
        phish_hints(path),


        count_dots(hostname),
        ratio_digits(hostname),
        

    ]
    return output

feature_extract_from_url("https://www.exampleurl.com/info/fuck/aboutus.html")
# print(len(feature_extract_from_url("https://www.exampleurl.com/info/aboutus.html")))


