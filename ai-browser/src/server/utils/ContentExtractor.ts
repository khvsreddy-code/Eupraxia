import { JSDOM } from 'jsdom';
import { Readability } from '@mozilla/readability';

export interface ExtractedContent {
  title: string;
  content: string;
  textContent: string;
  excerpt: string;
  siteName: string | null;
  author: string | null;
  timestamp: string | null;
}

export class ContentExtractor {
  static async extract(html: string, url: string): Promise<ExtractedContent> {
    const dom = new JSDOM(html, { url });
    const reader = new Readability(dom.window.document);
    const article = reader.parse();

    if (!article) {
      throw new Error('Failed to extract content');
    }

    // Get meta information
    const doc = dom.window.document;
    const siteName = doc.querySelector('meta[property="og:site_name"]')?.getAttribute('content') || 
                    doc.querySelector('meta[name="application-name"]')?.getAttribute('content') ||
                    null;

    const author = doc.querySelector('meta[name="author"]')?.getAttribute('content') ||
                  doc.querySelector('meta[property="article:author"]')?.getAttribute('content') ||
                  null;

    const timestamp = doc.querySelector('meta[property="article:published_time"]')?.getAttribute('content') ||
                     doc.querySelector('time')?.getAttribute('datetime') ||
                     null;

    return {
      title: article.title,
      content: article.content,
      textContent: article.textContent,
      excerpt: article.excerpt,
      siteName,
      author,
      timestamp
    };
  }

  static generateCitation(extracted: ExtractedContent, url: string): string {
    const date = extracted.timestamp ? new Date(extracted.timestamp).toLocaleDateString() : 'n.d.';
    const author = extracted.author || 'No author';
    const site = extracted.siteName || new URL(url).hostname;
    
    return `${author}. (${date}). ${extracted.title}. ${site}. Retrieved from ${url}`;
  }

  static summarize(content: string, maxLength: number = 250): string {
    // Simple summarization - take first paragraph or sentence
    const text = content.replace(/<[^>]*>/g, ''); // Remove HTML tags
    const sentences = text.split(/[.!?]+/);
    let summary = '';
    
    for (const sentence of sentences) {
      const trimmed = sentence.trim();
      if (!trimmed) continue;
      
      if ((summary + trimmed).length > maxLength) {
        return summary.trim() + '...';
      }
      summary += trimmed + '. ';
    }
    
    return summary.trim();
  }

  static extractKeywords(content: string): string[] {
    // Simple keyword extraction
    const text = content.toLowerCase().replace(/<[^>]*>/g, '');
    const words = text.split(/\W+/);
    const stopWords = new Set(['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at']);
    
    // Count word frequencies
    const frequencies = new Map<string, number>();
    for (const word of words) {
      if (word.length < 3 || stopWords.has(word)) continue;
      frequencies.set(word, (frequencies.get(word) || 0) + 1);
    }
    
    // Sort by frequency and return top 10
    return Array.from(frequencies.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([word]) => word);
  }
}