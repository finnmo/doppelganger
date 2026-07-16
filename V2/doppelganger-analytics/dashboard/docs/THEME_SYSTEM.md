# Theme System Documentation

## Overview

The Doppelganger Analytics Dashboard includes a comprehensive theme system that allows users to switch between different visual styles while maintaining consistent functionality. The system supports two primary themes:

- **Modern Theme**: Contemporary design with vibrant colors, gradients, and rounded corners
- **Classic Theme**: Professional appearance with muted colors and traditional styling

## Architecture

### Core Components

1. **ThemeContext** (`src/contexts/ThemeContext.tsx`)
   - Manages global theme state
   - Provides theme switching functionality  
   - Stores user preferences in localStorage
   - Defines theme configurations and class mappings

2. **ThemeSelector** (`src/components/ThemeSelector.tsx`)
   - User interface for theme selection
   - Dropdown with theme previews
   - Real-time theme switching

3. **Theme-Aware Components** (`src/components/ThemeAwareCard.tsx`)
   - Pre-built components that adapt to current theme
   - Consistent styling across the application
   - Multiple variants (hero, section, participant cards)

4. **CSS Variables** (`src/app/globals.css`)
   - Theme-specific color palettes
   - Utility classes for theme application
   - Smooth transitions between themes

## Usage Guide

### Basic Implementation

```tsx
import { useTheme } from '@/contexts/ThemeContext';
import { ThemeAwareCard } from '@/components/ThemeAwareCard';

function MyComponent() {
  const { themeStyle } = useTheme();
  
  return (
    <ThemeAwareCard variant="default" color="blue">
      <div className="p-6">
        <h3>Current theme: {themeStyle}</h3>
      </div>
    </ThemeAwareCard>
  );
}
```

### Theme-Aware Components

#### ThemeAwareCard
```tsx
<ThemeAwareCard 
  variant="default" | "hero" | "section" | "participant"
  color="blue" | "orange" | "green" | "purple" | "red" | "yellow" | "indigo" | "pink"
  className="additional-classes"
>
  {children}
</ThemeAwareCard>
```

#### ThemeAwareHeroCard
```tsx
<ThemeAwareHeroCard
  title="Total Messages"
  value={12543}
  icon={<MessageCircle className="w-6 h-6 mr-2" />}
  color="blue"
  details={[
    { label: "👥 Participants", value: 8 },
    { label: "😊 Emojis", value: 2341 }
  ]}
  footer="Across 5 conversations"
  tooltip={{
    title: "Total Messages",
    description: "Complete count of all messages",
    calculation: "Sum of all messages from participants",
    example: "1,000 + 2,500 + 1,200 = 4,700 messages"
  }}
/>
```

#### ThemeAwareParticipantCard
```tsx
<ThemeAwareParticipantCard
  title="Top Contributors"
  icon={<MessageCircle className="w-5 h-5 mr-2" />}
  color="blue"
  participants={[
    { name: "Alice Johnson", value: 3421, subtitle: "27%" },
    { name: "Bob Smith", value: 2987, subtitle: "24%" }
  ]}
  tooltip={{
    title: "Top Contributors",
    description: "Participants with most messages",
    calculation: "Sorted by message count",
    example: "Alice: 3,421 messages (27% of total)"
  }}
/>
```

### Manual Theme Application

For components that need custom theme handling:

```tsx
import { useTheme, themeClasses } from '@/contexts/ThemeContext';

function CustomComponent() {
  const { themeStyle } = useTheme();
  const classes = themeClasses[themeStyle];
  
  return (
    <div className={classes.card.default}>
      <div className={classes.text.primary}>
        Theme-aware content
      </div>
    </div>
  );
}
```

## Theme Configurations

### Modern Theme
- **Primary Color**: Blue (#3b82f6)
- **Secondary Color**: Green (#10b981)  
- **Accent Color**: Amber (#f59e0b)
- **Border Radius**: Large (rounded-2xl)
- **Shadows**: Enhanced depth
- **Gradients**: Vibrant color combinations

### Classic Theme
- **Primary Color**: Gray-800 (#1f2937)
- **Secondary Color**: Gray-600 (#374151)
- **Accent Color**: Gray-500 (#6b7280)
- **Border Radius**: Standard (rounded-lg)
- **Shadows**: Subtle
- **Gradients**: Minimal, professional

## CSS Variables

The system uses CSS custom properties for dynamic theming:

```css
:root {
  /* Modern Theme Colors */
  --modern-primary: #3b82f6;
  --modern-secondary: #10b981;
  --modern-accent: #f59e0b;
  
  /* Classic Theme Colors */
  --classic-primary: #1f2937;
  --classic-secondary: #374151;
  --classic-accent: #6b7280;
}

.theme-modern {
  --theme-primary: var(--modern-primary);
  --theme-secondary: var(--modern-secondary);
  --theme-accent: var(--modern-accent);
}

.theme-classic {
  --theme-primary: var(--classic-primary);
  --theme-secondary: var(--classic-secondary);
  --theme-accent: var(--classic-accent);
}
```

## Migration Guide

### Converting Existing Components

1. **Replace hardcoded colors** with theme-aware alternatives:
   ```tsx
   // Before
   <div className="bg-blue-500 text-white">
   
   // After  
   <ThemeAwareCard color="blue">
   ```

2. **Use theme context** for conditional styling:
   ```tsx
   // Before
   <div className="rounded-lg shadow-md">
   
   // After
   const { themeStyle } = useTheme();
   <div className={themeStyle === 'modern' ? 'rounded-2xl shadow-lg' : 'rounded-lg shadow-md'}>
   ```

3. **Leverage theme classes**:
   ```tsx
   // Before
   <div className="bg-white border border-gray-200">
   
   // After
   const classes = themeClasses[themeStyle];
   <div className={classes.card.default}>
   ```

## Best Practices

### 1. Consistent Component Usage
Always use theme-aware components when available rather than custom styling:
```tsx
// ✅ Good
<ThemeAwareCard variant="hero" color="blue">

// ❌ Avoid  
<div className="bg-blue-500 rounded-lg">
```

### 2. Proper Color Selection
Choose appropriate colors for different content types:
- **Blue**: Primary data, messages, communication
- **Orange**: Media, attachments, warnings
- **Green**: Success, growth, positive metrics
- **Purple**: Special features, emotions, reactions
- **Red**: Errors, urgent items

### 3. Responsive Design
Theme-aware components include responsive classes:
```tsx
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
  <ThemeAwareCard>Content</ThemeAwareCard>
</div>
```

### 4. Accessibility
Themes maintain proper contrast ratios and accessibility standards:
- Text remains readable in both themes
- Interactive elements have proper focus states
- Color is not the only means of conveying information

## Testing

### Theme Switching Tests
```tsx
import { render, screen } from '@testing-library/react';
import { ThemeProvider } from '@/contexts/ThemeContext';
import { ThemeDemo } from '@/components/ThemeDemo';

test('theme switching works correctly', () => {
  render(
    <ThemeProvider>
      <ThemeDemo />
    </ThemeProvider>
  );
  
  // Test theme selector presence
  expect(screen.getByText('Theme')).toBeInTheDocument();
  
  // Test component adaptation
  // Add specific tests for your components
});
```

### Visual Regression Testing
Consider implementing visual regression tests to ensure themes render correctly:
- Capture screenshots of key components in both themes
- Compare against baseline images
- Detect unintended visual changes

## Performance Considerations

### 1. CSS Variables
Using CSS custom properties allows for efficient theme switching without re-rendering components.

### 2. LocalStorage
Theme preferences are persisted to avoid flickering on page load:
```tsx
useEffect(() => {
  const savedTheme = localStorage.getItem('theme-style') as ThemeStyle;
  if (savedTheme) {
    setThemeStyle(savedTheme);
  }
}, []);
```

### 3. Component Memoization
Theme-aware components use React.memo where appropriate to prevent unnecessary re-renders.

## Extending the System

### Adding New Themes

1. **Define color palette** in `globals.css`:
   ```css
   :root {
     --dark-primary: #000000;
     --dark-secondary: #333333;
     /* ... */
   }
   
   .theme-dark {
     --theme-primary: var(--dark-primary);
     /* ... */
   }
   ```

2. **Update ThemeContext**:
   ```tsx
   export type ThemeStyle = 'modern' | 'classic' | 'dark';
   
   const themes: Record<ThemeStyle, ThemeConfig> = {
     modern: { /* ... */ },
     classic: { /* ... */ },
     dark: { /* ... */ }
   };
   ```

3. **Add to ThemeSelector**:
   ```tsx
   const themes = [
     { id: 'modern', name: 'Modern', /* ... */ },
     { id: 'classic', name: 'Classic', /* ... */ },
     { id: 'dark', name: 'Dark', /* ... */ }
   ];
   ```

### Custom Theme-Aware Components

```tsx
interface CustomThemeComponentProps {
  variant?: 'primary' | 'secondary';
  children: React.ReactNode;
}

export function CustomThemeComponent({ 
  variant = 'primary', 
  children 
}: CustomThemeComponentProps) {
  const { themeStyle } = useTheme();
  const classes = themeClasses[themeStyle];
  
  const getVariantClasses = () => {
    switch (variant) {
      case 'primary':
        return classes.custom.primary;
      case 'secondary':
        return classes.custom.secondary;
      default:
        return classes.custom.default;
    }
  };
  
  return (
    <div className={`${classes.card.default} ${getVariantClasses()}`}>
      {children}
    </div>
  );
}
```

## Troubleshooting

### Common Issues

1. **Theme not applying**: Ensure ThemeProvider wraps your app
2. **Flickering on load**: Check localStorage initialization
3. **Colors not updating**: Verify CSS variable names match
4. **Component not theme-aware**: Use themeClasses or theme-aware components

### Debug Mode
Enable debug logging in development:
```tsx
const ThemeContext = createContext<ThemeContextType>({
  // ... other properties
  debug: process.env.NODE_ENV === 'development'
});
```

## Future Enhancements

### Planned Features
- **System theme detection**: Auto-switch based on OS preference
- **Custom theme builder**: Allow users to create custom themes
- **Theme animations**: Smooth transitions between theme switches
- **Component theme overrides**: Per-component theme customization
- **Theme presets**: Pre-configured theme combinations

### Contributing
When adding new components:
1. Make them theme-aware by default
2. Follow established color and styling patterns
3. Include proper TypeScript interfaces
4. Add documentation and examples
5. Test in both themes

## Conclusion

The theme system provides a robust foundation for maintaining visual consistency while offering user customization. By following these guidelines and using the provided components, you can ensure your features integrate seamlessly with the overall design system.

For questions or contributions, please refer to the main project documentation or create an issue in the repository. 