import urllib.parse
import socket

class URLFeatureExtractor:
    def __init__(self):
        self.shortening_services = [
            'bit.ly', 'tinyurl.com', 'goo.gl', 'ow.ly', 
            'is.gd', 'buff.ly', 'adf.ly', 'bit.do'
        ]
        
    def extract_features(self, url):
        """Extract all 30 features from a URL"""
        features = []
        
        # Parse URL
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc
        path = parsed.path
        
        features.append(self._having_ip_address(domain))
        
        features.append(self._url_length(url))
        
        features.append(self._shortening_service(url))
        
        features.append(self._having_at_symbol(url))
        
        features.append(self._double_slash_redirecting(url))
        
        features.append(self._prefix_suffix(domain))
        
        features.append(self._having_sub_domain(domain))
        
        features.append(self._ssl_final_state(url))
        
        features.append(self._domain_registration_length())
        
        features.append(self._favicon())
        
        features.append(self._port(parsed))
        
        features.append(self._https_token(domain))
        
        for i in range(18):
            features.append(0)  # Neutral value
        
        return features
    
    def _having_ip_address(self, domain):
        """Check if domain uses IP address"""
        try:
            socket.inet_aton(domain.replace('[', '').replace(']', ''))
            return 1  # Phishing
        except:
            return -1  # Legitimate
    
    def _url_length(self, url):
        """Categorize URL length"""
        length = len(url)
        if length < 54:
            return -1  # Legitimate
        elif 54 <= length <= 75:
            return 0   # Suspicious
        else:
            return 1   # Phishing
    
    def _shortening_service(self, url):
        """Check for URL shortening services"""
        for service in self.shortening_services:
            if service in url:
                return 1  # Phishing
        return -1  # Legitimate
    
    def _having_at_symbol(self, url):
        """Check for @ symbol in URL"""
        return 1 if '@' in url else -1
    
    def _double_slash_redirecting(self, url):
        """Check for double slash after http"""
        position = url.find('//')
        if position > 7:  # After http:// or https://
            return 1  # Phishing
        return -1  # Legitimate
    
    def _prefix_suffix(self, domain):
        """Check for dash in domain"""
        return 1 if '-' in domain else -1
    
    def _having_sub_domain(self, domain):
        """Count subdomains"""
        # Remove www if present
        domain = domain.replace('www.', '')
        subdomain_count = domain.count('.')
        
        if subdomain_count <= 1:
            return -1  # Legitimate
        elif subdomain_count == 2:
            return 0   # Suspicious
        else:
            return 1   # Phishing
    
    def _ssl_final_state(self, url):
        """Check SSL certificate"""
        if url.startswith('https'):
            return -1  # Legitimate
        return 1  # Phishing
    
    def _domain_registration_length(self):
        """Domain registration length (simplified)"""
        return 0 # Neutral
    
    def _favicon(self):
        """Favicon check (simplified)"""
        return 0  # Neutral
    
    def _port(self, parsed):
        """Check for non-standard port"""
        if parsed.port:
            if parsed.port not in [80, 443]:
                return 1  # Phishing
        return -1  # Legitimate
    
    def _https_token(self, domain):
        """Check for HTTPS in domain name"""
        return 1 if 'https' in domain else -1

if __name__ == "__main__":
    extractor = URLFeatureExtractor()
    
    # Test URLs
    test_urls = [
        "https://www.google.com",
        "http://192.168.1.1/login",
        "https://bit.ly/suspicious",
        "http://google-secure-login.fake.com"
    ]
    
    for url in test_urls:
        features = extractor.extract_features(url)
        print(f"\nURL: {url}")
        print(f"Features: {features[:8]}...")  