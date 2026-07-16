// Global test setup
import fs from 'fs';
import path from 'path';

// Get directory path for tests - use process.cwd() for Jest compatibility
const testsDir = path.join(process.cwd(), 'tests');

// Extend Jest matchers
expect.extend({
  toBeValidJSON(received: string) {
    try {
      JSON.parse(received);
      return {
        message: () => `Expected ${received} not to be valid JSON`,
        pass: true,
      };
    } catch {
      return {
        message: () => `Expected ${received} to be valid JSON`,
        pass: false,
      };
    }
  },
  
  toHaveValidMetricStructure(received: any, expectedStructure: any) {
    const validateStructure = (obj: any, structure: any, path = ''): string[] => {
      const errors: string[] = [];
      
      for (const [key, expectedType] of Object.entries(structure)) {
        const currentPath = path ? `${path}.${key}` : key;
        
        if (!(key in obj)) {
          errors.push(`Missing property: ${currentPath}`);
          continue;
        }
        
        const actualValue = obj[key];
        
        if (typeof expectedType === 'string') {
          if (typeof actualValue !== expectedType) {
            errors.push(`Type mismatch at ${currentPath}: expected ${expectedType}, got ${typeof actualValue}`);
          }
        } else if (Array.isArray(expectedType)) {
          if (!Array.isArray(actualValue)) {
            errors.push(`Type mismatch at ${currentPath}: expected array, got ${typeof actualValue}`);
          } else if (expectedType.length > 0 && actualValue.length > 0) {
            // Validate first element structure
            errors.push(...validateStructure(actualValue[0], expectedType[0], `${currentPath}[0]`));
          }
        } else if (typeof expectedType === 'object' && expectedType !== null) {
          if (typeof actualValue !== 'object' || actualValue === null) {
            errors.push(`Type mismatch at ${currentPath}: expected object, got ${typeof actualValue}`);
          } else {
            errors.push(...validateStructure(actualValue, expectedType, currentPath));
          }
        }
      }
      
      return errors;
    };
    
    const errors = validateStructure(received, expectedStructure);
    
    return {
      message: () => `Metric structure validation failed:\n${errors.join('\n')}`,
      pass: errors.length === 0,
    };
  }
});

// Global test utilities
(global as any).testUtils = {
  loadTestFixture: (fixtureName: string) => {
    const fixturePath = path.join(testsDir, 'fixtures', `${fixtureName}.json`);
    if (fs.existsSync(fixturePath)) {
      return JSON.parse(fs.readFileSync(fixturePath, 'utf-8'));
    }
    throw new Error(`Test fixture not found: ${fixtureName}`);
  },
  
  loadTestFixtureMessages: () => {
    const fixturePath = path.join(testsDir, 'fixtures', 'basic-messages.json');
    const rawMessages = JSON.parse(fs.readFileSync(fixturePath, 'utf-8'));
    
    // Convert Instagram format to processed format with IDs
    return rawMessages.map((msg: any, index: number) => ({
      id: index + 1,
      conversation_id: 'test_conversation',
      sender: msg.sender_name,
      timestamp_ms: msg.timestamp_ms,
      content: msg.content || generateContentForMedia(msg),
      reply_to_message_id: null // Basic test data doesn't have replies
    }));
  },
  
  createMockMessages: (count: number = 10) => {
    const messages = [];
    const senders = ['alice', 'bob', 'charlie'];
    const baseTime = Date.now() - (24 * 60 * 60 * 1000); // 24 hours ago
    
    for (let i = 0; i < count; i++) {
      messages.push({
        id: i + 1,
        conversation_id: 'test_conversation',
        sender: senders[i % senders.length],
        timestamp_ms: baseTime + (i * 60 * 1000), // 1 minute apart
        content: `Test message ${i + 1}`,
        reply_to_message_id: i > 0 && Math.random() > 0.7 ? Math.floor(Math.random() * i) + 1 : null
      });
    }
    
    return messages;
  }
};

function generateContentForMedia(msg: any): string | null {
  if (msg.photos && msg.photos.length > 0) {
    return `${msg.sender_name} sent ${msg.photos.length} photo${msg.photos.length > 1 ? 's' : ''}`;
  }
  if (msg.videos && msg.videos.length > 0) {
    return `${msg.sender_name} sent ${msg.videos.length} video${msg.videos.length > 1 ? 's' : ''}`;
  }
  return null;
}

// Type declarations
declare global {
  namespace jest {
    interface Matchers<R> {
      toBeValidJSON(): R;
      toHaveValidMetricStructure(expectedStructure: any): R;
    }
  }
  
  var testUtils: {
    loadTestFixture: (fixtureName: string) => any;
    loadTestFixtureMessages: () => any[];
    createMockMessages: (count?: number) => any[];
  };
}

// Export for ES modules
export {}; 