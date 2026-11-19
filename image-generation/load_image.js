// A promptfoo provider that "returns" the image generate.py already saved, so promptfoo
// grades it without calling an image model. It reads the file named in the CSV's image
// column and outputs it as a data URL, which the judge and the results viewer can both show.
// (A file:// var would load the image too, but the viewer then shows a broken input image.)
const fs = require('fs');
const path = require('path');

module.exports = class SavedImage {
  id() {
    return 'saved-image';
  }

  async callApi(prompt, context) {
    const { image, mime } = context.vars;
    const data = fs.readFileSync(path.join(__dirname, image)).toString('base64');
    return { output: `data:${mime || 'image/png'};base64,${data}` };
  }
};
