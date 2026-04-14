const path = require('path');
const fs = require('fs');

module.exports = function pluginApiDocs(context) {
  return {
    name: 'plugin-api-docs',
    async loadContent() {
      const opsPath = path.join(context.siteDir, 'static', 'ops.json');
      const raw = fs.readFileSync(opsPath, 'utf8');
      return JSON.parse(raw);
    },
    async contentLoaded({content, actions}) {
      const {setGlobalData} = actions;
      setGlobalData(content);
    },
  };
};
