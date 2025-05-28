# TAAT Visual Design System and Style Guide

## 1. Introduction

This document establishes the visual design system and style guide for the TAAT (Twitter Trade Announcer Tool) application. It provides comprehensive guidelines for visual elements, components, and patterns to ensure a consistent, accessible, and professional user experience across all platforms and devices.

## 2. Brand Identity

### 2.1 Brand Values

The TAAT visual identity reflects the following brand values:

- **Trustworthy**: Instills confidence through professional design and clarity
- **Efficient**: Optimizes for quick comprehension and action
- **Intelligent**: Communicates sophistication and data-driven insights
- **Responsive**: Adapts seamlessly to user needs and different contexts
- **Accessible**: Ensures usability for all users regardless of abilities

### 2.2 Logo and Wordmark

The TAAT logo combines a stylized "T" with a trading chart motif, representing the bridge between social media signals and trading actions.

- **Primary Logo**: Full-color version for standard use
- **Monochrome Logo**: Single-color version for limited color applications
- **Favicon/App Icon**: Simplified version for small displays
- **Minimum Size**: 24px height to maintain legibility
- **Clear Space**: Maintain padding of at least 50% of logo height on all sides

## 3. Color Palette

### 3.1 Primary Colors

- **TAAT Blue** (#1A73E8)
  - Main brand color
  - Used for primary actions, links, and key UI elements
  - Dark variant: #0D47A1
  - Light variant: #BBDEFB

- **TAAT Green** (#34A853)
  - Positive indicators, success states, and upward trends
  - Dark variant: #0F9D58
  - Light variant: #C8E6C9

- **TAAT Red** (#EA4335)
  - Negative indicators, error states, and downward trends
  - Dark variant: #B31412
  - Light variant: #FFCDD2

### 3.2 Neutral Colors

- **Dark Gray** (#202124)
  - Primary text color
  - Dark backgrounds

- **Medium Gray** (#5F6368)
  - Secondary text color
  - Borders and dividers

- **Light Gray** (#DADCE0)
  - Subtle UI elements
  - Disabled states

- **Off-White** (#F8F9FA)
  - Page backgrounds
  - Card backgrounds

### 3.3 Accent Colors

- **TAAT Yellow** (#FBBC04)
  - Warnings and cautions
  - Highlighting important information

- **TAAT Purple** (#9334E8)
  - Premium features
  - Advanced analytics

### 3.4 Semantic Colors

- **Success** (#34A853) - Same as TAAT Green
- **Warning** (#FBBC04) - Same as TAAT Yellow
- **Error** (#EA4335) - Same as TAAT Red
- **Info** (#1A73E8) - Same as TAAT Blue

### 3.5 Color Usage Guidelines

- **Contrast Ratios**: Maintain minimum contrast ratios of 4.5:1 for normal text and 3:1 for large text (WCAG AA compliance)
- **Color Combinations**: Predefined combinations for various UI elements
- **Color Meaning**: Consistent use of colors for specific meanings (e.g., green for positive, red for negative)
- **Dark Mode**: Color adaptations for dark mode theme

## 4. Typography

### 4.1 Font Family

- **Primary Font**: Inter
  - Clean, modern sans-serif typeface optimized for screen readability
  - Web font and system fallbacks defined

- **Monospace Font**: Roboto Mono
  - Used for code, data values, and technical information
  - Web font and system fallbacks defined

### 4.2 Type Scale

- **Display**: 40px/48px, Inter Medium
- **Heading 1**: 32px/40px, Inter Medium
- **Heading 2**: 24px/32px, Inter Medium
- **Heading 3**: 20px/28px, Inter Medium
- **Heading 4**: 16px/24px, Inter Medium
- **Body Large**: 16px/24px, Inter Regular
- **Body**: 14px/20px, Inter Regular
- **Body Small**: 12px/16px, Inter Regular
- **Caption**: 11px/16px, Inter Regular
- **Button**: 14px/20px, Inter Medium
- **Data**: 14px/20px, Roboto Mono Regular

### 4.3 Typography Guidelines

- **Line Length**: Maximum 75 characters per line for optimal readability
- **Alignment**: Left-aligned text for most content (RTL languages will be right-aligned)
- **Emphasis**: Use weight (Medium) rather than italics for emphasis
- **Case**: Sentence case for headings and labels
- **Truncation**: Ellipsis for truncated text with full text available on hover

## 5. Spacing System

### 5.1 Base Unit

The spacing system is built on a base unit of 4px, creating a consistent rhythm throughout the interface.

### 5.2 Spacing Scale

- **2xs**: 4px (1× base unit)
- **xs**: 8px (2× base unit)
- **sm**: 12px (3× base unit)
- **md**: 16px (4× base unit)
- **lg**: 24px (6× base unit)
- **xl**: 32px (8× base unit)
- **2xl**: 48px (12× base unit)
- **3xl**: 64px (16× base unit)

### 5.3 Layout Spacing

- **Page Margins**: 24px (desktop), 16px (tablet), 16px (mobile)
- **Card Padding**: 24px (desktop), 16px (mobile)
- **Section Spacing**: 48px between major sections
- **Component Spacing**: 24px between components
- **Element Spacing**: 16px between related elements
- **Content Spacing**: 8px between content items

## 6. Component Library

### 6.1 Core Components

#### 6.1.1 Buttons

- **Primary Button**
  - Background: TAAT Blue
  - Text: White
  - Height: 40px
  - Padding: 8px 24px
  - Border Radius: 4px
  - States: Default, Hover, Active, Focus, Disabled

- **Secondary Button**
  - Background: White
  - Border: 1px solid TAAT Blue
  - Text: TAAT Blue
  - Height: 40px
  - Padding: 8px 24px
  - Border Radius: 4px
  - States: Default, Hover, Active, Focus, Disabled

- **Tertiary Button**
  - Background: Transparent
  - Text: TAAT Blue
  - Height: 40px
  - Padding: 8px 16px
  - Border Radius: 4px
  - States: Default, Hover, Active, Focus, Disabled

- **Icon Button**
  - Size: 40px × 40px
  - Border Radius: 20px (circular)
  - States: Default, Hover, Active, Focus, Disabled

#### 6.1.2 Form Elements

- **Text Input**
  - Height: 40px
  - Padding: 8px 12px
  - Border: 1px solid Light Gray
  - Border Radius: 4px
  - States: Default, Focus, Error, Disabled

- **Dropdown**
  - Height: 40px
  - Padding: 8px 12px
  - Border: 1px solid Light Gray
  - Border Radius: 4px
  - States: Default, Open, Focus, Disabled

- **Checkbox**
  - Size: 16px × 16px
  - States: Unchecked, Checked, Indeterminate, Focus, Disabled

- **Radio Button**
  - Size: 16px × 16px
  - States: Unselected, Selected, Focus, Disabled

- **Toggle Switch**
  - Size: 32px × 16px
  - States: Off, On, Focus, Disabled

#### 6.1.3 Cards

- **Standard Card**
  - Background: White
  - Border Radius: 8px
  - Shadow: 0 2px 4px rgba(0,0,0,0.1)
  - Padding: 24px
  - States: Default, Hover, Active

- **Signal Card**
  - Background: White
  - Border-left: 4px solid (varies by signal type)
  - Border Radius: 8px
  - Shadow: 0 2px 4px rgba(0,0,0,0.1)
  - Padding: 16px
  - States: Default, Hover, Active

- **Dashboard Widget**
  - Background: White
  - Border Radius: 8px
  - Shadow: 0 2px 4px rgba(0,0,0,0.1)
  - Padding: 16px
  - Header: 16px padding, bottom border
  - Body: 16px padding
  - Footer: 16px padding, top border

#### 6.1.4 Navigation

- **Top Navigation Bar**
  - Height: 64px
  - Background: White
  - Shadow: 0 2px 4px rgba(0,0,0,0.1)
  - Active Indicator: 2px bottom border in TAAT Blue

- **Side Navigation**
  - Width: 240px (desktop), collapsible to 64px
  - Background: White
  - Item Height: 48px
  - Active Indicator: 4px left border in TAAT Blue

- **Tab Bar**
  - Height: 48px
  - Background: White
  - Active Indicator: 2px bottom border in TAAT Blue

- **Mobile Bottom Navigation**
  - Height: 56px
  - Background: White
  - Shadow: 0 -2px 4px rgba(0,0,0,0.1)
  - Active Indicator: Color change to TAAT Blue

### 6.2 Specialized Components

#### 6.2.1 Data Visualization

- **Charts and Graphs**
  - Color Scheme: Primary and accent colors with semantic meaning
  - Grid Lines: Light Gray, 1px
  - Axes: Medium Gray, 1px
  - Labels: Body Small typography
  - Tooltips: Card style with 8px padding

- **Data Tables**
  - Header: Light Gray background, Medium Gray text
  - Rows: Alternating White and Off-White
  - Borders: Light Gray, 1px
  - Padding: 12px
  - Sorting Indicators: Small arrows next to column headers

#### 6.2.2 Trading Components

- **Signal Indicator**
  - New: 4px left border in TAAT Blue
  - Pending: 4px left border in TAAT Yellow
  - Completed: 4px left border in TAAT Green
  - Rejected: 4px left border in Medium Gray

- **Price Display**
  - Positive: TAAT Green
  - Negative: TAAT Red
  - Neutral: Dark Gray
  - Format: $XXX.XX (+/-X.X%)

- **Order Status**
  - Pending: TAAT Yellow pill
  - Executed: TAAT Green pill
  - Rejected: TAAT Red pill
  - Canceled: Medium Gray pill

#### 6.2.3 Notifications

- **Toast Notification**
  - Background: Dark Gray
  - Text: White
  - Border Radius: 4px
  - Padding: 12px 16px
  - Duration: 5 seconds

- **Alert Banner**
  - Success: Light Green background, TAAT Green border
  - Warning: Light Yellow background, TAAT Yellow border
  - Error: Light Red background, TAAT Red border
  - Info: Light Blue background, TAAT Blue border
  - Padding: 12px 16px
  - Border Radius: 4px

- **Badge**
  - Size: 16px × 16px
  - Border Radius: 8px (circular)
  - Background: TAAT Red (or other semantic colors)
  - Text: White, Caption typography

## 7. Iconography

### 7.1 Icon System

- **Style**: Outlined, consistent 2px stroke weight
- **Size**: 24px × 24px (standard), 16px × 16px (small)
- **Grid**: 24px × 24px base grid with 1px increments
- **Padding**: Minimum 2px padding from edge

### 7.2 Icon Categories

- **Navigation Icons**: Home, signals, portfolio, trading, settings
- **Action Icons**: Add, edit, delete, share, download
- **Status Icons**: Success, warning, error, info
- **Trading Icons**: Buy, sell, limit, market, stop
- **Chart Icons**: Line, bar, candlestick, area

### 7.3 Icon Usage Guidelines

- **Consistency**: Use the same icon for the same action across the application
- **Text Labels**: Pair icons with text labels when possible
- **Color**: Use icon color to convey state or importance
- **Touch Targets**: Minimum 44px × 44px touch target for interactive icons

## 8. Responsive Design

### 8.1 Breakpoints

- **Mobile**: 320px - 599px
- **Tablet**: 600px - 1023px
- **Desktop**: 1024px - 1439px
- **Large Desktop**: 1440px and above

### 8.2 Layout Grid

- **Columns**: 4 (mobile), 8 (tablet), 12 (desktop)
- **Gutters**: 16px (mobile), 24px (tablet), 24px (desktop)
- **Margins**: 16px (mobile), 24px (tablet), 24px (desktop)

### 8.3 Responsive Patterns

- **Stacking**: Multi-column layouts stack vertically on smaller screens
- **Reflow**: Content reflows to fit available space
- **Progressive Disclosure**: Less critical information is hidden behind expandable sections on smaller screens
- **Touch Optimization**: Larger touch targets on touch devices

## 9. Motion and Animation

### 9.1 Timing

- **Quick**: 100ms - For immediate feedback (button presses)
- **Standard**: 200ms - For typical transitions
- **Elaborate**: 300ms - For more complex animations

### 9.2 Easing

- **Standard**: Ease-out (fast start, slow end)
- **Decelerate**: Cubic-bezier(0.0, 0.0, 0.2, 1)
- **Accelerate**: Cubic-bezier(0.4, 0.0, 1, 1)

### 9.3 Animation Patterns

- **Page Transitions**: Fade and slight movement
- **Component Entry/Exit**: Fade combined with scale or movement
- **Loading States**: Pulsing or spinning indicators
- **Data Updates**: Subtle highlighting of changed values

### 9.4 Reduced Motion

- Alternative animations for users with reduced motion preferences
- Static indicators instead of animations where appropriate

## 10. Accessibility Guidelines

### 10.1 Color and Contrast

- **Text Contrast**: Minimum 4.5:1 for normal text, 3:1 for large text
- **UI Component Contrast**: Minimum 3:1 for interactive elements
- **Non-Color Indicators**: Always pair color with another indicator (icon, text, pattern)

### 10.2 Typography Accessibility

- **Minimum Text Size**: 12px for body text
- **Line Height**: Minimum 1.5× font size for body text
- **Font Weight**: Avoid using weights below Regular (400) for body text

### 10.3 Keyboard Navigation

- **Focus Indicators**: Visible focus state for all interactive elements
- **Focus Order**: Logical tab order following visual layout
- **Keyboard Shortcuts**: Documented keyboard shortcuts for common actions

### 10.4 Screen Reader Support

- **Semantic HTML**: Use appropriate HTML elements for their intended purpose
- **ARIA Labels**: Provide descriptive labels for custom components
- **Alternative Text**: Descriptive alt text for all images and icons
- **Live Regions**: Use ARIA live regions for dynamic content updates

## 11. Implementation Guidelines

### 11.1 CSS Architecture

- **Naming Convention**: BEM (Block, Element, Modifier)
- **CSS Variables**: Use for colors, typography, and spacing
- **Responsive Utilities**: Mobile-first approach with breakpoint utilities

### 11.2 Component Implementation

- **React Components**: Implement as React components with TypeScript
- **Storybook**: Document components in Storybook with examples and props
- **Testing**: Include accessibility and visual regression tests

### 11.3 Design Token System

- **Format**: JSON format for design tokens
- **Categories**: Colors, typography, spacing, shadows, borders
- **Platforms**: Web (CSS variables), mobile (if applicable)

## 12. Asset Management

### 12.1 Image Guidelines

- **Formats**: SVG for icons and illustrations, WebP with PNG fallback for photos
- **Optimization**: Compress all images appropriately
- **Resolution**: Support for standard and high-DPI displays

### 12.2 File Organization

- **Naming Convention**: kebab-case for all asset files
- **Directory Structure**: Organized by component and asset type
- **Version Control**: All design assets in version control

## 13. Design System Governance

### 13.1 Contribution Process

- **Proposals**: Process for proposing new components or modifications
- **Review**: Design review process for additions to the system
- **Documentation**: Requirements for documenting new components

### 13.2 Version Control

- **Semantic Versioning**: Follow semver for design system releases
- **Changelog**: Document all changes in each release
- **Deprecation**: Process for deprecating and removing components

## 14. Resources

### 14.1 Design Files

- **Figma Library**: Link to Figma design system library
- **Component Specifications**: Detailed specifications for each component

### 14.2 Code Resources

- **Component Library**: Link to React component library
- **Storybook**: Link to Storybook documentation
- **GitHub Repository**: Link to design system code repository

## 15. Appendix

### 15.1 Glossary

- Definitions of key terms used throughout the design system

### 15.2 Version History

- Record of major updates to the design system

---

This visual design system and style guide will evolve as the TAAT application develops. All team members should refer to this document when designing, developing, or modifying the user interface to ensure consistency and quality across the application.
