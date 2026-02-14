#!/usr/bin/env python3
"""
Système de Surveillance Épidémiologique - Tanger-Tétouan-Al Hoceima
Scanne les sources d'actualités pour détecter les signaux de santé publique
Version 100% GRATUITE avec Groq (Llama 3.3 70B - 30 req/min gratuit)
"""

import os
import json
import yaml
import requests
import feedparser
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import time

class HealthSurveillanceBot:
    def __init__(self, config_path='config.yaml'):
        """Initialize le bot avec la config"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.telegram_token = os.environ.get('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        self.groq_key = os.environ.get('GROQ_API_KEY')
        
        if not all([self.telegram_token, self.telegram_chat_id, self.groq_key]):
            raise ValueError("❌ Variables d'environnement manquantes!")
        
        self.articles_seen = self.load_seen_articles()
    
    def load_seen_articles(self):
        """Charge les articles déjà vus (pour éviter les doublons)"""
        try:
            with open('seen_articles.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def save_seen_articles(self):
        """Sauvegarde les articles vus"""
        with open('seen_articles.json', 'w') as f:
            json.dump(self.articles_seen, f)
    
    def fetch_rss_feed(self, feed_url):
        """Récupère les articles d'un flux RSS"""
        try:
            feed = feedparser.parse(feed_url)
            articles = []
            
            for entry in feed.entries[:10]:  # Limite à 10 articles par source
                # Vérifie si l'article est récent (dernières 48h)
                pub_date = entry.get('published_parsed') or entry.get('updated_parsed')
                if pub_date:
                    article_date = datetime(*pub_date[:6])
                    if datetime.now() - article_date > timedelta(days=2):
                        continue
                
                article = {
                    'title': entry.get('title', 'Sans titre'),
                    'link': entry.get('link', ''),
                    'description': entry.get('summary', entry.get('description', '')),
                    'source': feed_url,
                    'published': pub_date
                }
                articles.append(article)
            
            return articles
        except Exception as e:
            print(f"⚠️ Erreur RSS {feed_url}: {e}")
            return []
    
    def analyze_with_groq(self, articles):
        """Analyse les articles avec Groq (Llama 3.3 70B - GRATUIT)"""
        if not articles:
            return []
        
        # Prépare le prompt avec tous les articles
        articles_text = "\n\n---\n\n".join([
            f"ARTICLE {i+1}\nTitre: {a['title']}\nSource: {a['source']}\nLien: {a['link']}\nContenu: {a['description'][:500]}"
            for i, a in enumerate(articles)
        ])
        
        prompt = f"""Tu es un expert en surveillance épidémiologique type EIOS/WHO pour la région Tanger-Tétouan-Al Hoceima au Maroc.

CONTEXTE EIOS:
- Détection précoce de signaux (event-based surveillance)
- Approche One Health (humain-animal-environnement)
- Focus sur signaux inhabituels/émergents
- Surveillance transfrontalière (Ceuta/Melilla, Algérie)

MISSION: Analyser ces articles et identifier UNIQUEMENT les signaux épidémiologiques pertinents.

CRITÈRES EIOS (signal quality assessment):
✓ Nouveauté: nouveau pathogène, symptômes atypiques, zone inhabituelle
✓ Gravité: cas sévères, décès, hospitalisation
✓ Agrégation: clusters, cas groupés, augmentation rapide
✓ Vulnérabilité: populations à risque (enfants, personnes âgées, immunodéprimés)
✓ Potentiel épidémique: transmission rapide, contagiosité
✓ One Health: zoonoses, contamination environnementale, santé animale
✓ Contexte régional: pays limitrophes, flux transfrontaliers

SIGNAUX PRIORITAIRES:
- Maladies à potentiel épidémique (méningite, choléra, rougeole, etc.)
- Syndromes respiratoires/fébriles inhabituels
- Toxi-infections alimentaires collectives (TIAC)
- Zoonoses émergentes
- Mortalité/morbidité inexpliquée
- Alertes pays limitrophes (Espagne/Algérie)

EXCLUS (bruit informationnel):
✗ Politiques sanitaires / inaugurations
✗ Campagnes de sensibilisation
✗ Statistiques annuelles
✗ Cas individuels isolés sans contexte
✗ Maladies chroniques (diabète, cancer) sauf cluster inhabituel

ARTICLES À ANALYSER:
{articles_text}

RÉPONDS UNIQUEMENT EN JSON, format:
{{
  "articles_pertinents": [
    {{
      "article_numero": 1,
      "niveau_risque": "faible|moyen|élevé|critique",
      "raison": "Justification EIOS du signal (pourquoi c'est pertinent)",
      "type_signal": "cluster|émergent|zoonose|transfrontalier|TIAC|autre",
      "mots_cles": ["mot1", "mot2", "mot3"]
    }}
  ]
}}

Si aucun signal pertinent: {{"articles_pertinents": []}}"""

        try:
            # Appel API Groq (GRATUIT - 30 req/min)
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",  # Meilleur modèle gratuit
                    "messages": [
                        {"role": "system", "content": "Tu es un expert en surveillance épidémiologique EIOS/WHO. Tu réponds UNIQUEMENT en JSON valide."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,  # Précision maximale
                    "max_tokens": 2000
                },
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Parse la réponse JSON
            response_text = result['choices'][0]['message']['content'].strip()
            
            # Nettoie le JSON si nécessaire
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            analysis = json.loads(response_text)
            
            # Mappe les résultats aux articles originaux
            relevant_articles = []
            for item in analysis.get('articles_pertinents', []):
                idx = item['article_numero'] - 1
                if 0 <= idx < len(articles):
                    article = articles[idx].copy()
                    article['risk_level'] = item['niveau_risque']
                    article['reason'] = item['raison']
                    article['signal_type'] = item.get('type_signal', 'autre')
                    article['keywords'] = item['mots_cles']
                    relevant_articles.append(article)
            
            return relevant_articles
            
        except Exception as e:
            print(f"❌ Erreur analyse Groq: {e}")
            return []
    
    def send_telegram_message(self, message):
        """Envoie un message via Telegram"""
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        data = {
            'chat_id': self.telegram_chat_id,
            'text': message,
            'parse_mode': 'HTML',
            'disable_web_page_preview': False
        }
        
        try:
            response = requests.post(url, data=data)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"❌ Erreur Telegram: {e}")
            return False
    
    def format_report(self, articles):
        """Formate le rapport pour Telegram (style EIOS)"""
        if not articles:
            return "🟢 <b>Surveillance EIOS-style - Tanger-Tétouan-Al Hoceima</b>\n\n✅ Aucun signal épidémiologique détecté aujourd'hui."
        
        # Trie par niveau de risque
        risk_order = {'critique': 0, 'élevé': 1, 'moyen': 2, 'faible': 3}
        articles.sort(key=lambda x: risk_order.get(x.get('risk_level', 'faible'), 99))
        
        # Emojis par niveau
        risk_emoji = {
            'critique': '🔴',
            'élevé': '🟠',
            'moyen': '🟡',
            'faible': '🟢'
        }
        
        # Type de signal emoji
        signal_emoji = {
            'cluster': '👥',
            'émergent': '🆕',
            'zoonose': '🦠',
            'transfrontalier': '🌍',
            'TIAC': '🍽️',
            'autre': '⚠️'
        }
        
        report = f"<b>🏥 Event-Based Surveillance - {datetime.now().strftime('%d/%m/%Y')}</b>\n"
        report += f"<b>📍 Région: Tanger-Tétouan-Al Hoceima</b>\n\n"
        report += f"<b>🚨 {len(articles)} signal(aux) épidémiologique(s)</b>\n\n"
        report += "━━━━━━━━━━━━━━━━\n\n"
        
        for i, article in enumerate(articles, 1):
            emoji = risk_emoji.get(article.get('risk_level', 'faible'), '⚪')
            sig_emoji = signal_emoji.get(article.get('signal_type', 'autre'), '⚠️')
            
            report += f"{emoji} <b>Signal #{i}</b> {sig_emoji} {article.get('signal_type', 'N/A').upper()}\n"
            report += f"<b>Niveau: {article.get('risk_level', 'N/A').upper()}</b>\n\n"
            report += f"<b>📰 {article['title']}</b>\n\n"
            report += f"💡 <i>{article.get('reason', 'N/A')}</i>\n\n"
            report += f"🔗 <a href='{article['link']}'>Lire l'article</a>\n\n"
            
            if article.get('keywords'):
                report += f"🏷️ {', '.join(article['keywords'])}\n\n"
            
            report += "━━━━━━━━━━━━━━━━\n\n"
        
        report += f"<i>Powered by Groq AI | Surveillance style EIOS</i>"
        
        return report
    
    def run_surveillance(self):
        """Exécute le scan complet"""
        print(f"🔍 Démarrage surveillance EIOS-style - {datetime.now()}")
        
        all_articles = []
        
        # Parcourt toutes les sources
        for source in self.config['sources']:
            print(f"📡 Scan: {source['name']}")
            
            if source['type'] == 'rss':
                articles = self.fetch_rss_feed(source['url'])
            else:
                continue  # Skip autres types pour l'instant
            
            # Filtre les articles déjà vus
            new_articles = [
                a for a in articles 
                if a['link'] not in self.articles_seen
            ]
            
            all_articles.extend(new_articles)
            print(f"   ✓ {len(new_articles)} nouveaux articles")
        
        print(f"\n📊 Total: {len(all_articles)} articles à analyser")
        
        if not all_articles:
            message = "🟢 <b>Surveillance EIOS-style - TTA</b>\n\n✅ Aucun nouvel article aujourd'hui."
            self.send_telegram_message(message)
            return
        
        # Analyse avec Groq (par batches de 15 articles - rate limit)
        relevant_articles = []
        batch_size = 15
        
        for i in range(0, len(all_articles), batch_size):
            batch = all_articles[i:i+batch_size]
            print(f"🤖 Analyse Groq batch {i//batch_size + 1} ({len(batch)} articles)...")
            
            results = self.analyze_with_groq(batch)
            relevant_articles.extend(results)
            
            # Marque tous les articles comme vus
            for article in batch:
                self.articles_seen[article['link']] = datetime.now().isoformat()
            
            time.sleep(2)  # Rate limiting Groq (30 req/min gratuit)
        
        # Sauvegarde les articles vus
        self.save_seen_articles()
        
        print(f"✅ {len(relevant_articles)} signaux épidémiologiques détectés")
        
        # Envoie le rapport
        report = self.format_report(relevant_articles)
        
        # Telegram limite à 4096 caractères
        if len(report) > 4000:
            # Divise en plusieurs messages
            parts = []
            current = ""
            for line in report.split('\n'):
                if len(current) + len(line) > 3900:
                    parts.append(current)
                    current = line + '\n'
                else:
                    current += line + '\n'
            if current:
                parts.append(current)
            
            for part in parts:
                self.send_telegram_message(part)
                time.sleep(1)
        else:
            self.send_telegram_message(report)
        
        print("✅ Rapport envoyé!")

if __name__ == "__main__":
    bot = HealthSurveillanceBot()
    bot.run_surveillance()
