/*
 * Copyright 2016 - 2024 Drew Edwards, tmpim
 *
 * This file is part of Krist.
 *
 * Krist is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Krist is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Krist. If not, see <http://www.gnu.org/licenses/>.
 *
 * For more project information, see <https://github.com/tmpim/krist>.
 */

import dayjs from "dayjs";
import { Router } from "express";
import { ctrlGetTransaction, ctrlGetTransactions, ctrlMakeTransaction } from "../../controllers/transactions.js";
import { getRecentTransactions, transactionToJson } from "../../krist/transactions/index.js";
import { padDigits } from "../../utils/index.js";
import { PaginatedQuery, ReqQuery, returnPaginatedResult } from "../utils.js";

/**
 * @apiDefine TransactionGroup Transactions
 *
 * All Transactions related endpoints.
 */

/**
 * @apiDefine Transaction
 *
 * @apiSuccess {Object} transaction
 * @apiSuccess {Number} transaction.id The ID of this transaction.
 * @apiSuccess {String} transaction.from The sender of this transaction. This
 *   may be `null`.
 * @apiSuccess {String} transaction.to The recipient of this transaction. This
 *   may be `"name"` if the transaction was a name purchase, or `"a"` if it was
 *   a name's data change.
 * @apiSuccess {Number} transaction.value The amount of Bacon transferred in
 *   this transaction. Can be `0`, notably if the transaction was a name's data
 *   change.
 * @apiSuccess {Date} transaction.time The time this transaction this was made,
 *   as an ISO-8601 string.
 * @apiSuccess {String} [transaction.name] The name associated with this
 *   transaction, without the `.bcn` suffix, or `null`.
 * @apiSuccess {String} [transaction.metadata] Transaction metadata, or `null`.
 * @apiSuccess {String} [transaction.sent_metaname] The metaname (part before
 *   the `"@"`) of the recipient of this transaction, if it was sent to a name.
 * @apiSuccess {String} [transaction.sent_name] The name this transaction was
 *   sent to, without the `.bcn` suffix, if it was sent to a name.
 * @apiSuccess {String} transaction.type The type of this transaction. May be
 *   `mined`, `transfer`, `name_purchase`, `name_a_record`, or `name_transfer`.
 *   Note that `name_a_record` refers to a name's data changing.
 */

/**
 * @apiDefine Transactions
 *
 * @apiSuccess {Object[]} transactions
 * @apiSuccess {Number} transactions.id The ID of this transaction.
 * @apiSuccess {String} transactions.from The sender of this transaction. This
 *   may be `null`.
 * @apiSuccess {String} transactions.to The recipient of this transaction. This
 *   may be `"name"` if the transaction was a name purchase, or `"a"` if it was
 *   a name's data change.
 * @apiSuccess {Number} transactions.value The amount of Bacon transferred in
 *   this transaction. Can be `0`, notably if the transaction was a name's data
 *   change.
 * @apiSuccess {Date} transactions.time The time this transaction this was made,
 *   as an ISO-8601 string.
 * @apiSuccess {String} [transactions.name] The name associated with this
 *   transaction, without the `.bcn` suffix, or `null`.
 * @apiSuccess {String} [transactions.metadata] Transaction metadata, or `null`.
 * @apiSuccess {String} [transactions.sent_metaname] The metaname (part before
 *   the `"@"`) of the recipient of this transaction, if it was sent to a name.
 * @apiSuccess {String} [transactions.sent_name] The name this transaction was
 *   sent to, without the `.bcn` suffix, if it was sent to a name.
 * @apiSuccess {String} transactions.type The type of this transaction. May be
 *   `mined`, `transfer`, `name_purchase`, `name_a_record`, or `name_transfer`.
 *   Note that `name_a_record` refers to a name's data changing.
  */

export default (): Router => {
  const router = Router();

  // ===========================================================================
  // API v2
  // ===========================================================================
  /**
	 * @api {get} /transactions List all transactions
	 * @apiName GetTransactions
	 * @apiGroup TransactionGroup
	 * @apiVersion 2.3.0
	 *
	 * @apiUse LimitOffset
	 * @apiQuery {Boolean} [excludeMined] If specified,
	 *   transactions from mining will be excluded.
	 *
	 * @apiSuccess {Number} count The count of results.
	 * @apiSuccess {Number} total The total amount of transactions.
	 * @apiUse Transactions
	 *
	 * @apiSuccessExample {json} Success
	 * {
   *     "ok": true,
   *     "count": 50,
   *     "total": 175000,
   *     "transactions": [
   *         {
   *             "id": 1,
   *             "from": null,
   *             "to": "0000000000",
   *             "value": 50,
   *             "time": "2015-02-14T16:44:40.000Z",
   *             "name": null,
   *             "metadata": null,
   *             "sent_metaname": null,
   *             "sent_name": null,
   *             "type": "mined"
   *         },
   *         {
   *             "id": 46,
   *             "from": "71fd7571b9",
   *             "to": "a5dfb396d3",
   *             "value": 1000,
   *             "time": "2015-02-14T23:15:39.000Z",
   *             "name": null,
   *             "metadata": null,
   *             "sent_metaname": null,
   *             "sent_name": null,
   *             "type": "transfer"
   *         },
	 *  	   ...
	 */
  router.get("/transactions", async (req: PaginatedQuery<{
    excludeMined?: string;
  }>, res) => {
    const results = await ctrlGetTransactions(
      req.query.limit, req.query.offset,
      true,
      req.query.excludeMined === undefined
    );

    returnPaginatedResult(res, "transactions", transactionToJson, results);
  });

  /**
	 * @api {get} /transactions/latest List latest transactions
	 * @apiName GetLatestTransactions
	 * @apiGroup TransactionGroup
	 * @apiVersion 2.3.0
	 *
	 * @apiUse LimitOffset
	 * @apiQuery {Boolean} [excludeMined] If specified,
	 *   transactions from mining will be excluded.
	 *
	 * @apiSuccess {Number} count The count of results.
	 * @apiSuccess {Number} total The total amount of transactions.
	 * @apiUse Transactions
	 *
	 * @apiSuccessExample {json} Success
	 * {
   *     "ok": true,
   *     "count": 50,
   *     "total": 175000,
   *     "transactions": [
   *         {
   *             "id": 153287,
   *             "from": null,
   *             "to": "b70dxcdr3n",
   *             "value": 14,
   *             "time": "2016-02-06T19:22:41.000Z",
   *             "name": null,
   *             "metadata": null,
   *             "sent_metaname": null,
   *             "sent_name": null,
   *             "type": "mined"
   *         },
   *         {
   *             "id": 153286,
   *             "from": "bxxhsp1uzh",
   *             "to": "name",
   *             "value": 500,
   *             "time": "2016-02-06T14:01:19.000Z",
   *             "name": "exam",
   *             "metadata": null,
   *             "sent_metaname": null,
   *             "sent_name": null,
   *             "type": "name_purchase"
   *         },
	 *  	   ...
	 */
  router.get("/transactions/latest", async (req: PaginatedQuery<{
    excludeMined?: string;
  }>, res) => {
    const results = await ctrlGetTransactions(
      req.query.limit, req.query.offset,
      false,
      req.query.excludeMined === undefined
    );

    returnPaginatedResult(res, "transactions", transactionToJson, results);
  });

  /**
	 * @api {get} /transactions/:id Get a transaction
	 * @apiName GetTransaction
	 * @apiGroup TransactionGroup
	 * @apiVersion 2.3.0
	 *
	 * @apiParam {Number} id The ID of the transaction to get.
	 *
	 * @apiUse Transaction
	 *
	 * @apiSuccessExample {json} Success
	 * {
   *     "ok": true,
   *     "transaction": {
   *         "id": 153282,
   *         "from": "bh9w36ea1b",
   *         "to": "butlg1kzhz",
   *         "value": 56610,
   *         "time": "2016-02-03T19:15:32.000Z",
   *         "name": null,
   *         "metadata": null,
   *         "sent_metaname": null,
   *         "sent_name": null,
   *         "type": "transfer"
   *     }
   * }
	 */
  router.get("/transactions/:id", async (req, res) => {
    const tx = await ctrlGetTransaction(req.params.id);
    res.json({
      ok: true,
      transaction: transactionToJson(tx)
    });
  });

  /**
	 * @api {post} /transactions/ Make a transaction
	 * @apiName MakeTransaction
	 * @apiGroup TransactionGroup
	 * @apiVersion 2.0.0
	 *
	 * @apiBody {String} privatekey The privatekey of your address.
	 * @apiBody {String} to The recipient of the transaction.
	 * @apiBody {Number} amount The amount to send to the recipient.
	 * @apiBody {String} [metadata] Optional metadata to include in the transaction.
   * @apiBody {String} [requestId] Optional request ID to use for this transaction. Must be a valid UUIDv4 if provided.
	 *
	 * @apiUse Transaction
	 *
	 * @apiSuccessExample {json} Success
	 * {
   *     "ok": true
   * }
	 *
	 * @apiErrorExample {json} Insufficient Funds
	 * {
   *     "ok": false,
   *     "error": "insufficient_funds"
   * }
   *
   * @apiErrorExample {json} Transaction Conflict
   * {
   *     "ok": false,
   *     "error": "transaction_conflict",
   *     "parameter": "amount"
   * }
	 */
  router.post("/transactions", async (req, res) => {
    const tx = await ctrlMakeTransaction(
      req,
      req.body.privatekey,
      req.body.to,
      req.body.amount,
      req.body.metadata,
      req.body.requestId
    );

    res.json({
      ok: true,
      transaction: transactionToJson(tx)
    });
  });

  // ===========================================================================
  // Legacy API
  // ===========================================================================
  router.get("/", async (req: ReqQuery<{
    recenttx?: string;
    pushtx?: string;
    pushtx2?: string;
    pkey?: string;
    q?: string;
    amt?: string;
    com?: string;
  }>, res, next) => {
    if (req.query.recenttx !== undefined) {
      const { rows } = await getRecentTransactions();

      const lines = rows.map(tx =>
        dayjs(tx.time).format("MMM DD HH:mm")
        + tx.from
        + tx.to
        + padDigits(Math.abs(tx.value), 8));

      return res.send(lines.join(""));
    }

    if (req.query.pushtx !== undefined) {
      return res.send("v1 transactions disabled. Contact Bacon team");
    }

    if (req.query.pushtx2 !== undefined) {
      return res.send("Legacy transactions disabled. Contact Bacon team");
    }

    next();
  });

  return router;
};
