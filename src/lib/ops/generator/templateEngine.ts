export type TemplateFunction = (symbolMap: {[key: string]: string | undefined}, config?: {[key: string]: boolean}) => string;

const enum TokenType {
    TEXT,
    IF,
    ELSE,
    ELSEIF,
    ENDIF,
    IFNOT
}

interface Token {
    type: TokenType; // 0 - normal, 1 - #if, 2 - #else, 3 - #elseif, 4 - #endif, 5 - #ifnot
    value: string;
}

/**
 * A very basic template engine.
 */
export class TemplateEngine {

    private _cache: {[key: string]: TemplateFunction} = {};

    public generate(template: string,
                    symbolMap: {[key: string]: string | undefined},
                    config?: {[key: string]: boolean}): string {
        if (!this._cache[template]) {
            // create a new generator
            this._cache[template] = this.createFunction(template);
        }
        let gen = this._cache[template];
        return gen(symbolMap, config);
    }

    public createFunction(template: string): TemplateFunction {
        let condReg = /^[ \t]*#(if|endif|else|elseif|ifnot)([ \t]+(\w+)[ \t]*)?$/gm;
        let tokens: Token[] = [];
        let idx = 0;
        // tokenize
        while (idx < template.length) {
            let match = condReg.exec(template);
            if (match) {
                if (match.index > idx) {
                    tokens.push({
                        type: TokenType.TEXT,
                        value: TemplateEngine._sanitize(template.slice(idx, match.index))
                    });
                }
                switch (match[1]) {
                    case 'if':
                        if (!match[3]) {
                            throw new Error('Missing condition after if.');
                        }
                        tokens.push({
                            type: TokenType.IF,
                            value: match[3]
                        });
                        break;
                    case 'else':
                        if (match[3]) {
                            throw new Error('Unexpected condition after else.');
                        }
                        tokens.push({
                            type: TokenType.ELSE,
                            value: ''
                        });
                        break;
                    case 'elseif':
                            if (!match[3]) {
                            throw new Error('Missing condition after elseif.');
                        }
                        tokens.push({
                            type: TokenType.ELSEIF,
                            value: match[3]
                        });
                        break;
                    case 'ifnot':
                        if (!match[3]) {
                            throw new Error('Missing condition after ifnot.');
                        }
                        tokens.push({
                            type: TokenType.IFNOT,
                            value: match[3]
                        });
                        break;
                    case 'endif':
                        if (match[3]) {
                            throw new Error('Unexpected condition after endif.');
                        }
                        tokens.push({
                            type: TokenType.ENDIF,
                            value: ''
                        });
                        break;
                    default:
                        throw new Error(`Unknown keyword '${match[1]}'.`);
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
            switch (tokens[i].type) {
                case TokenType.TEXT:
                    funcBody += `result += '${tokens[i].value}';\n`;
                    break;
                case TokenType.IF:
                    funcBody += `if (config.${tokens[i].value}) {\n`;
                    balanceCounter++;
                    break;
                case TokenType.IFNOT:
                    funcBody += `if (!config.${tokens[i].value}) {\n`;
                    balanceCounter++;
                    break;
                case TokenType.ELSE:
                    funcBody += `} else {\n`;
                    break;
                case TokenType.ELSEIF:
                    funcBody += `} else if (config.${tokens[i].value}) {\n`
                    break;
                case TokenType.ENDIF:
                    funcBody += '}\n';
                    balanceCounter--;
                    break;
                default:
                    throw new Error(`Unexpected token type ${tokens[i].type}.`);
            }
        }
        funcBody += 'return result;';
        if (balanceCounter !== 0) {
            throw new Error('Unbalanced if and endif statements.');
        }
        let templateFunc = new Function('config', funcBody);
        return (symbolMap, config) => {
            return TemplateEngine._interpolate(
                templateFunc(config),
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