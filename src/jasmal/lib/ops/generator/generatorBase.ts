import { TemplateEngine } from './templateEngine';

export abstract class OpGeneratorBase {

    /**
     * Name of the object used to inject dependencies.
     * In the generated functions, the dependencies are accessed via
     * `__dep__.Dependency`.
     */
    public readonly DEP_OBJ_NAME = '__dep__';

    protected _engine: TemplateEngine;

    protected constructor() {
        this._engine = new TemplateEngine();
    }

    /**
     * Checks the symbols used in the given code. Throws when encounters any
     * symbol that is not allowed. Returns a list of used symbols.
     * @param code Block of code to be checked.
     * @param allowed A list of allowed symbols. Case sensitive.
     */
    protected _checkUsedSymbols(code: string, allowed: string[]): string[] {
        let reSymbol = /\$\w+/g;
        let m: RegExpExecArray | null;
        let used: {[key: string]: boolean} = {};
        while (m = reSymbol.exec(code)) {
            if (allowed.indexOf(m[0]) < 0) {
                throw new Error(`Symbol ${m[0]} is not permitted in the following code:\n${code}`);
            }
            used[m[0]] = true;
        }
        let result: string[] = [];
        for (let prop in used) {
            if (used.hasOwnProperty(prop)) {
                result.push(prop);
            }
        }
        return result;
    }

    /**
     * Converts inline functions to string.
     * @param fs Functions to be converted.
     */
    protected _flattenInlineFunctions(fs: {[key: string]: Function}): string {
        let result = '';
        for (let key in fs) {
            if (fs.hasOwnProperty(key)) {
                let fStr = fs[key].toString();
                if (fStr.indexOf('[native code]') > 0) {
                    throw new Error('Cannot inline native functions.');
                }
                // replace function name
                let idxFirstP = fStr.indexOf('(');
                if (idxFirstP < 0) {
                    throw new Error('Cannot find the first pair of parenthesis.');
                }
                // note that the specified function name is NOT checked
                fStr = 'function ' + key + fStr.substr(idxFirstP);
                result += fStr + '\n';
            }
        }
        return result;
    }

    /**
     * Generate the code block to import dependencies.
     * @param depNames Dependency names (accessible from depObjName).
     * @example
     *  _generateDependencyBlock('__dep__', ['Core', 'Utils']);
     * // var Core = __dep__.Core;
     * // var Utils = __dep__.Utils;
     */
    protected _generateDependencyBlock(depNames: string[]): string {
        return depNames.map(x => `var ${x} = ${this.DEP_OBJ_NAME}.${x};`).join('\n');
    }
    
}

