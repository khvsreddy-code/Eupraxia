export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
}

export class ChatContextManager {
  private static instance: ChatContextManager;
  private messages: Message[] = [];
  private maxContextSize: number = 10;
  private systemPrompt: string = 'You are a helpful AI assistant with access to web browsing and advanced visualization capabilities.';

  private constructor() {
    this.messages.push({
      role: 'system',
      content: this.systemPrompt,
      timestamp: Date.now()
    });
  }

  static getInstance(): ChatContextManager {
    if (!ChatContextManager.instance) {
      ChatContextManager.instance = new ChatContextManager();
    }
    return ChatContextManager.instance;
  }

  public addMessage(role: Message['role'], content: string): void {
    this.messages.push({
      role,
      content,
      timestamp: Date.now()
    });

    // Trim context if it exceeds max size
    if (this.messages.length > this.maxContextSize + 1) { // +1 for system prompt
      // Keep system prompt and remove oldest messages
      const systemPrompt = this.messages[0];
      this.messages = [systemPrompt, ...this.messages.slice(-this.maxContextSize)];
    }
  }

  public getContext(): Message[] {
    return [...this.messages];
  }

  public getRecentMessages(count: number): Message[] {
    return this.messages.slice(-count);
  }

  public clearContext(): void {
    const systemPrompt = this.messages[0];
    this.messages = [systemPrompt];
  }

  public updateSystemPrompt(newPrompt: string): void {
    this.systemPrompt = newPrompt;
    this.messages[0] = {
      role: 'system',
      content: newPrompt,
      timestamp: Date.now()
    };
  }

  public async getContextEmbeddings(): Promise<number[][]> {
    // TODO: Implement embeddings generation
    // This will be used for semantic search and context relevance
    return [];
  }

  public summarizeContext(): string {
    // Create a brief summary of the conversation
    const messageCount = this.messages.length - 1; // Exclude system prompt
    const userMessages = this.messages.filter(m => m.role === 'user').length;
    const assistantMessages = this.messages.filter(m => m.role === 'assistant').length;
    
    return `Conversation with ${messageCount} messages (${userMessages} user, ${assistantMessages} assistant)`;
  }
}