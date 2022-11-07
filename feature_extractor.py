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
        raw_words = list(filter(None, raw_words))
        return raw_words, list(filter(None, w_host)), list(filter(None, w_path))

    #subdomain_name = 3,   hostname = 3 + 2 + 1
    hostname, second_level_domain, path = get_domain(url)
    
    #domain_name = 2 + 1
    extracted_domain = tldextract.extract(url)
    domain_name = extracted_domain.domain + '.' + extracted_domain.suffix

    subdomain = extracted_domain.subdomain
    words_raw, words_raw_host, words_raw_path = words_raw_extraction(domain_name, subdomain, path[1:])

    top_level_domain = extracted_domain.suffix
    parsed = urlparse(url)
    scheme = parsed.scheme

    output = [
        #url-based
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

        #word raws based features
        check_www(words_raw),
        check_com(words_raw),
        length_word_raw(words_raw),
        char_repeat(words_raw),
        shortest_word_length(words_raw),
        shortest_word_length(words_raw_host),
        shortest_word_length(words_raw_path),
        longest_word_length(words_raw),
        longest_word_length(words_raw_host),
        longest_word_length(words_raw_path),
        average_word_length(words_raw),
        average_word_length(words_raw_host),
        average_word_length(words_raw_path),

        #domain-based
        tld_in_path(top_level_domain, path),
        tld_in_subdomain(top_level_domain, subdomain),
        tld_in_bad_position(top_level_domain, subdomain, path),
        suspecious_tld(top_level_domain),
        domain_in_brand(second_level_domain),
        brand_in_path(second_level_domain, subdomain),
        # statistical_report(url, domain_name),# 有問題

        #scheme-based
        https_token(scheme)
    ]
    return output

# feature_extract_from_url("https://www.exampleurl.com/info/fuck/aboutus.html")
# print(feature_extract_from_url("https://www.exampleurl.com/info/aboutus.html"))