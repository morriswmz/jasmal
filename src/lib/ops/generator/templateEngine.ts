export type Generator = (symbolMap: {[key: string]: string | undefined}, config?: {[key: string]: boolean}) => string;

interface Token {
    type: number; // 0 - normal, 1 - #if, 2 - #else, 3 - #elseif, 4 - #endif, 5 - #ifnot
    value: string;
}

/**
 * A very basic template engine.
 */
export class TemplateEngine {

    private _cache: {[key: string]: Generator} = {};

    public generate(template: string,
                    symbolMap: {[key: string]: string | undefined},
                    config?: {[key: string]: boolean}): string {
        if (!this._cache[template]) {
            // create a new generator
            this._cache[template] = this.createGenerator(template);
        }
        let gen = this._cache[template];
        return gen(symbolMap, config);
    }

    public createGenerator(template: string): Generator {
        let condReg = /^[ \t]*#(if|endif|else|elseif|ifnot)([ \t]+(\w+)[ \t]*)?$/gm;
        let tokens: Token[] = [];
        let idx = 0;
        // tokenize
        while (idx < template.length) {
            let match = condReg.exec(template);
            if (match) {
                if (match.index > idx) {
                    tokens.push({
                        type: 0,
                        value: TemplateEngine._sanitize(template.slice(idx, match.index))
                    });
                }
                if (match[1] === 'if') {
                    if (!match[3]) {
                        throw new Error('Missing condition after if.');
                    }
                    tokens.push({
                        type: 1,
                        value: match[3]
                    });
                } else if (match[1] === 'ifnot') {
                    if (!match[3]) {
                        throw new Error('Missing condition after if.');
                    }
                    tokens.push({
                        type: 5,
                        value: match[3]
                    });
                } else if (match[1] === 'else') {
                    if (match[3]) {
                        throw new Error('Unexpected condition after else.');
                    }
                    tokens.push({
                        type: 2,
                        value: ''
                    });
                } else if (match[1] === 'elseif') {
                    if (!match[3]) {
                        throw new Error('Missing condition after elseif.');
                    }
                    tokens.push({
                        type: 3,
                        value: match[3]
                    });
                } else {
                    if (match[3]) {
                        throw new Error('Unexpected condition after endif.');
                    }
                    tokens.push({
                        type: 4,
                        value: ''
                    });
                }
                idx = match.index + match[0].length;
                while (idx < template.length && (template[idx] === '\r' || template[idx] === '\n')) {
                    idx++;
                }
            } else {
                break;
            }
        }
        if (idx < template.length) {
            tokens.push({
                type: 0,
                value: TemplateEngine._sanitize(template.slice(idx, template.length))
            });
        }
        // process tokens
        let balanceCounter = 0;
        let funcBody = `var result = '';\n`;
        for (let i = 0;i < tokens.length;i++) {
            if (tokens[i].type === 0) {
                funcBody += `result += '${tokens[i].value}';\n`;
            } else if (tokens[i].type === 1) {
                funcBody += `if (config.${tokens[i].value}) {\n`;
                balanceCounter++;
            } else if (tokens[i].type === 2) {
                funcBody += `} else {\n`;
            } else if (tokens[i].type === 3) {
                funcBody += `} else if (config.${tokens[i].value}) {\n`
            } else if (tokens[i].type === 4) {
                funcBody += '}\n';
                balanceCounter--;
            } else if (tokens[i].type === 5) {
                funcBody += `if (!config.${tokens[i].value}) {\n`;
                balanceCounter++;
            } else {
                throw new Error('Unexpected token id.');
            }
        }
        funcBody += 'return result;';
        if (balanceCounter !== 0) {
            throw new Error('Unbalanced if and endif statements.');
        }
        let templateCompiler = new Function('config', funcBody);
        return (symbolMap, config) => {
            return TemplateEngine._interpolate(
                templateCompiler(config),
                symbolMap
            );
        };
    }

    private static _sanitize(str: string): string {
        return str.replace(/\\/g, '\\\\').replace(/'/g, '\\\'').replace(/(\r\n)|\r|\n/g, '\\n');
    }

    private static _interpolate(template: string, replacementMap: {[key: string]: string | undefined}): string {
        return template.replace(/\$\w+/g, (m, offset: number) => {
            // Check if we need to insert indentations.
            // Only checks for white spaces here and tabs are not allowed.
            let standalone = true;
            let pos = offset - 1;
            let indent = 0;
            while (pos >= 0) {
                if (template[pos] === '\n') {
                    break;
                }
                if (template[pos] !== ' ') {
                    standalone = false;
                    break;
                }
                indent++;
                pos--;
            }
            if (standalone) {
                pos = offset + m.length;
                while (pos < template.length) {
                    if (template[pos] === '\n') {
                        break;
                    }
                    if (template[pos] !== ' ') {
                        standalone = false;
                    }
                    pos++;
                }
            }
            let replacement = replacementMap[m];
            if (replacement != undefined) {
                return standalone ? TemplateEngine._indent(replacement, indent, false) : replacement;
            } else {
                return m;
            }         
        });
    }

     private static _indent(str: string, indentSize: number, indentFirstLine: boolean = true): string {
        // a little bit lazy here
        let spaces = (new Array<Number>(indentSize + 1)).join(' ');
        let result = str.replace(/(\r?\n)/g, '$1' + spaces);
        return indentFirstLine ? spaces + result : result;
    }

}