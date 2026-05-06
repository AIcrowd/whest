import '../app/global.css';

/** @type { import('@storybook/react').Preview } */
const preview = {
  parameters: {
    actions: { argTypesRegex: '^on[A-Z].*' },
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/i,
      },
    },
    backgrounds: {
      default: 'paper',
      values: [
        { name: 'paper', value: '#FFFFFF' },
        { name: 'soft', value: '#F8F9F9' },
        { name: 'gridLine', value: '#F4F6F6' },
      ],
    },
    layout: 'padded',
  },
};

export default preview;
