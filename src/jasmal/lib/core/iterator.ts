// Collections of index iterators for tensor slicing.
// ==================================================

export interface IIndexIterator {
    /**
     * Checks if there is any additional index.
     */
    hasNext(): boolean;
    /**
     * Retrieves the next available index and advance the iterator.
     */
    next(): number;
    /**
     * Retrieves the next available index without advancing the iterator.
     */
    peekNext(): number;
    /**
     * Resets the iterator to its beginning.
     */
    reset(): void;
    /**
     * Retrieves the total number of indices available.
     */
    count: number;
}

export abstract class IndexIterator implements IIndexIterator {

    protected _current: number;

    
    public hasNext(): boolean {
        throw new Error('Not implemented.');
    }

    public next(): number {
        throw new Error('Not implemented.');
    }

    public peekNext(): number {
        if (this.hasNext()) {
            return this._current;
        } else {
            throw new Error('Already reached the end of the iterator.');
        }
    }

    public get count(): number {
        throw new Error('Not implemented.');
    }

    public reset(): void {
        throw new Error('Not implemented.');
    }

}

export class ConstantIndexIterator extends IndexIterator {

    private _visited: boolean = false;

    constructor(index: number)
    {
        super();
        this._current = index;
    }

    public hasNext(): boolean {
        return !this._visited;
    }

    public next(): number {
        if (this.hasNext()) {
            this._visited = true;
            return this._current;
        } else {
            throw new Error('Already reached the end of the iterator.');
        }
    }

    public reset(): void {
        this._visited = false;
    }

    public get count(): number {
        return 1;
    }

}

export class RangedIndexIterator extends IndexIterator {

    protected _start: number;
    protected _stop: number;
    protected _step: number;

    constructor(start: number, stop: number, step: number) {
        super();
        if (step <= 0) {
            throw new Error('Step size must be positive.');
        }
        this._start = start;
        this._stop = stop;
        this._step = step;
        this._current = start;
    }

    public hasNext(): boolean {
        return this._current < this._stop;
    }

    public next(): number {
        if (this.hasNext()) {
            let cur = this._current;
            this._current += this._step;
            return cur;
        } else {
            throw new Error('Already reached the end of the iterator.');
        }
    }

    public reset(): void {
        this._current = this._start;
    }

    public get count(): number {
        if (this._start >= this._stop) {
            return 0;
        } else {
            return Math.floor((this._stop - this._start - 1) / this._step) + 1;
        }
    }

}

export class ReversedRangedIndexIterator extends RangedIndexIterator {

    protected _start: number;
    protected _stop: number;
    protected _step: number;

    public hasNext(): boolean {
        return this._current > this._stop;
    }

    public next(): number {
        if (this.hasNext()) {
            let cur = this._current;
            this._current -= this._step;
            return cur;
        } else {
            throw new Error('Already reached the end of the iterator.');
        }
    }

    public get count(): number {
        if (this._start <= this._stop) {
            return 0;
        } else {
            return Math.floor((this._start - this._stop - 1) / this._step) + 1;
        }
    }

}

export class ArrayBasedIndexIterator extends IndexIterator {

    private _indices: ArrayLike<number>;

    constructor(indices: ArrayLike<number>) {
        super();
        this._indices = indices;
        this._current = 0;
    }
    
    public hasNext(): boolean {
        return this._current < this._indices.length;
    }

    public next(): number {
        if (this.hasNext()) {
            return this._indices[this._current++];
        } else {
            throw new Error('Already reached the end of the iterator.');
        }
    }

    public peekNext(): number {
        if (this.hasNext()) {
            return this._indices[this._current];
        } else {
            throw new Error('Already reached the end of the iterator.');
        }
    }

    public reset(): void {
        this._current = 0;
    }

    public get count(): number {
        return this._indices.length;
    }

}
